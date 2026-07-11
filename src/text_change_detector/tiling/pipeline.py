from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.signal import find_peaks

from text_change_detector.shared.embedder import Embedder, SentenceTransformerEmbedder
from text_change_detector.shared.models import Community, Segment, SemanticUnit, TilingResult
from text_change_detector.shared.graph import knn_sparsify
from text_change_detector.tiling.extraction import Extractor, PdfReader, Source, builtin_extract


def tile(
    source: Source,
    *,
    extractor: Extractor | None = None,
    spacy_model: str | None = None,
    pdf_reader: PdfReader | None = None,
    embedder: Embedder | None = None,
    model_name: str | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = torch.float16,
    batch_size: int = 8,
    group_max_len: int = 7,
    window_size: int = 4,
    prominence_c: float = 1.0,
    min_solo_words: int = 10,
    knn_k: int = 5,
    louvain_seed: int = 0,
) -> TilingResult:
    """Tile a document into semantic units grouped into communities.

    Resolves `source` into a list of `Segment`s, segments them into semantic units
    with embedding-based text tiling, then clusters the units into communities
    on a kNN similarity graph.

    The source is resolved in this order:
        - a list[Segment] is used as-is (you parsed the document yourself);
        - a custom `extractor` is called as `extractor(path)`;
        - otherwise a built-in extractor is picked from the source: a
          python-docx Document or a *.docx path use the .docx extractor; a
          *.pdf path uses the .pdf extractor, which requires `pdf_reader`.
          Built-in extraction requires `spacy_model`.

    Args:
        source: A path to a .docx/.pdf file, an already-loaded python-docx
            Document, or a list[Segment] you produced yourself.
        extractor: Custom extractor turning a path into a list of `Segment`s; any
            `Callable[[Path], list[Segment]]`. Ignored when `source` is a
            list[Segment].
        spacy_model: spaCy model for the built-in extractor (e.g.
            "en_core_web_sm"), installed with `python -m spacy download <model>`.
            Required only when a built-in extractor runs; ignored otherwise.
        pdf_reader: Reading strategy for PDF sources; any `PdfReader` turning a
            PDF path into a list of `Block`s. The core ships no PDF engine, so a
            reader must be injected from a companion package (the pypdfium2
            rewrite or the PyMuPDF adapter). Required only when a *.pdf path is
            extracted by the built-in extractor; ignored otherwise.
        embedder: Custom embedding model; any object exposing
            `encode(list[str], normalize_embeddings=True) -> np.ndarray`. When
            None, a default SentenceTransformer is built, owned and freed by
            the library.
        model_name: Model id for the default embedder, ignored when `embedder`
            is given. None uses the default embedder's own model.
        device: Torch device for the default embedder; None auto-detects cuda/cpu.
        dtype: Torch dtype for the default embedder.
        batch_size: Encode batch size for the default embedder. The default of
            8 keeps the default SentenceTransformer from running out of memory
            on a 16 GB GPU even for very large documents.
        group_max_len: Maximum number of segments grouped into one unit.
        window_size: Sentences on each side used to score step dissimilarity.
        prominence_c: Cutoff for the prominence boundary test, applied as
            mean - prominence_c * std over the peak prominences of the
            dissimilarity signal. Lower keeps only the sharpest boundaries;
            by about 1.0 every prominence peak counts as a boundary.
        min_solo_words: A single-segment unit shorter than this keeps growing instead
            of standing alone.
        knn_k: Neighbours kept per unit when building the similarity graph.
        louvain_seed: Seed for Louvain community detection.

    Returns:
        TilingResult: the communities, each holding its semantic units.

    Note:
        With the default embedder the library owns it and frees its GPU memory
        before returning. With a custom `embedder`, its lifecycle is yours.
    """
    segments = _extract(source, extractor, spacy_model, pdf_reader)

    return _tile(
        segments, embedder, model_name, device, dtype, batch_size,
        group_max_len, window_size, prominence_c, min_solo_words, knn_k, louvain_seed,
    )


def _extract(
    source: Source,
    extractor: Extractor | None,
    spacy_model: str | None,
    pdf_reader: PdfReader | None = None,
) -> list[Segment]:
    if isinstance(source, list):
        return source

    if isinstance(source, str):
        source = Path(source)

    if extractor is not None:
        return extractor(source)

    return builtin_extract(source, spacy_model, pdf_reader)


def _tile(
    segments: list[Segment],
    embedder: Embedder | None,
    model_name: str | None,
    device: str | None,
    dtype: torch.dtype | None,
    batch_size: int | None,
    group_max_len: int,
    window_size: int,
    prominence_c: float,
    min_solo_words: int,
    knn_k: int,
    louvain_seed: int,
) -> TilingResult:
    owns_embedder = embedder is None

    if owns_embedder:
        kwargs = {"device": device, "dtype": dtype, "batch_size": batch_size}

        if model_name is not None:
            kwargs["model_name"] = model_name

        embedder = SentenceTransformerEmbedder(**kwargs)

    try:
        groups = _build_groups(
            segments, embedder, group_max_len, window_size, prominence_c, min_solo_words
        )
        unique = _deduplicate_groups(groups)
        matrix = _create_similarity_matrix(unique, embedder)
        adjacency = knn_sparsify(matrix, knn_k)
        graph = nx.from_numpy_array(adjacency)
        communities = nx.community.louvain_communities(graph, seed=louvain_seed)

        return _to_result(unique, communities)
    finally:
        if owns_embedder:
            embedder.close()


def _build_groups(
    segments: list[Segment],
    embedder: Embedder,
    group_max_len: int = 7,
    window_size: int = 4,
    prominence_c: float = 1.0,
    min_solo_words: int = 10,
) -> list[list[Segment]]:
    d = _step_dissimilarities([segment.text for segment in segments], embedder, window_size)
    boundaries = _prominence_boundaries(d, prominence_c)

    def cut_at(gap: int) -> bool:
        return 0 <= gap < len(d) and gap in boundaries

    def force_grow(current_window: deque[Segment]) -> bool:
        return len(current_window) == 1 and len(current_window[0].text.split()) < min_solo_words

    groups = []
    i = 0

    while i < len(segments):
        current_window = deque(maxlen=group_max_len)

        current_window.append(segments[i])

        left_idx = i
        right_idx = i
        left_closed = False
        right_closed = False

        while len(current_window) < group_max_len and (not left_closed or not right_closed):
            if not left_closed:
                gap = left_idx - 1

                if gap < 0:
                    left_closed = True
                elif cut_at(gap) and not force_grow(current_window):
                    left_closed = True
                else:
                    current_window.appendleft(segments[left_idx - 1])

                    left_idx -= 1

            if len(current_window) >= group_max_len:
                break

            if not right_closed:
                gap = right_idx

                if gap > len(segments) - 2:
                    right_closed = True
                elif cut_at(gap) and not force_grow(current_window):
                    right_closed = True
                else:
                    current_window.append(segments[right_idx + 1])

                    right_idx += 1

        groups.append(list(current_window))

        i = right_idx + 1

    return groups


def _step_dissimilarities(sentences: list[str], embedder: Embedder, window_size: int = 4) -> np.ndarray:
    n = len(sentences)
    left = [" ".join(sentences[max(0, g - window_size + 1):g + 1]) for g in range(n - 1)]
    right = [" ".join(sentences[g + 1:g + 1 + window_size]) for g in range(n - 1)]
    left_emb = embedder.encode(left, normalize_embeddings=True)
    right_emb = embedder.encode(right, normalize_embeddings=True)
    sims = np.sum(left_emb * right_emb, axis=1)

    return 1.0 - sims


def _prominence_boundaries(d: np.ndarray, c: float = 1.0) -> set[int]:
    peaks, properties = find_peaks(d, prominence=0)

    if len(peaks) == 0:
        return set()

    prominences = properties["prominences"]
    cutoff = prominences.mean() - c * prominences.std()

    return {int(peak) for peak, prom in zip(peaks, prominences) if prom >= cutoff}


def _deduplicate_groups(groups: list[list[Segment]]) -> list[list[Segment]]:
    flat = [_unit_text(g) for g in groups]
    result = []
    seen = set()

    for i, g in enumerate(flat):
        if any(g != other and g in other for other in flat):
            continue

        if g in seen:
            continue

        seen.add(g)
        result.append(groups[i])

    return result


def _create_similarity_matrix(groups: list[list[Segment]], embedder: Embedder) -> np.ndarray:
    flat = [_unit_text(g) for g in groups]
    embeddings = embedder.encode(flat, normalize_embeddings=True)
    matrix = embeddings @ embeddings.T

    return matrix


def _unit_text(group: list[Segment]) -> str:
    return " ".join(part for segment in group for part in (segment.text, *segment.payload))


def _to_result(unique_groups: list[list[Segment]], communities: list[set[int]]) -> TilingResult:
    return TilingResult(
        communities=[
            Community(
                id=i,
                units=[
                    SemanticUnit(
                        id=int(n),
                        section=unique_groups[n][0].section,
                        sentences=[segment.text for segment in unique_groups[n]],
                        payload=[p for segment in unique_groups[n] for p in segment.payload],
                    )
                    for n in sorted(c)
                ],
            )
            for i, c in enumerate(communities)
        ]
    )
