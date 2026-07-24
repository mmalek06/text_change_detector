from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from text_change_detector.detection.llm import (
    ChatModel,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_RETRIES,
    Reviewer,
    default_llm,
)
from text_change_detector.detection.models import (
    Change,
    ChangeImpact,
    DetectionResult,
    Relation,
    Suggestion,
)
from text_change_detector.detection.prompts import ENGLISH_PROMPTS, Prompts
from text_change_detector.shared.embedder import Embedder, SentenceTransformerEmbedder
from text_change_detector.shared.graph import knn_sparsify
from text_change_detector.shared.models import TilingResult


def detect_changes(
    tiling: TilingResult,
    changes: Iterable[Change | dict],
    *,
    embedder: Embedder | None = None,
    model_name: str | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = torch.float16,
    batch_size: int = 8,
    llm: ChatModel | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    prompts: Prompts = ENGLISH_PROMPTS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    requests_per_minute: int | None = None,
    max_concurrency: int = 1,
    knn_k: int = 5,
    primary_k: int = 5,
    similarity_floor: float = 0.5,
) -> DetectionResult:
    """Detect which semantic units each proposed change impacts.

    Rebuilds the same kNN relation graph from a `TilingResult`, ranks the units
    each change resembles (direct impact) plus their one-hop neighbours (ripple),
    then has an LLM rate every candidate, verify the strong ones with a skeptical
    second pass, and draft a minimal merged text for those that survive.

    Args:
        tiling: The semantic-units graph to analyse, as returned by `tile()`.
        changes: The proposed new or changed requirements to test. Each item is a
            `Change` (or a dict with `name` and `text`). This is always supplied
            by the caller; the library ships no changes of its own.
        embedder: Custom embedding model exposing
            `encode(list[str], normalize_embeddings=True) -> np.ndarray`. Must be
            the same model the graph was tiled with, so similarities line up. When
            None, a default SentenceTransformer is built, owned and freed by the
            library before the LLM pass runs (so both can share one GPU).
        model_name: Model id for the default embedder, ignored when `embedder`
            is given. None uses the default embedder's own model.
        device: Torch device for the default embedder; None auto-detects cuda/cpu.
        dtype: Torch dtype for the default embedder.
        batch_size: Encode batch size for the default embedder.
        llm: A LangChain chat model to run the passes on; anything supporting
            `with_structured_output(schema).invoke(prompt)`. When None, a default
            local `ChatOllama(model=llm_model)` is built.
        llm_model: Ollama model id for the default LLM, ignored when `llm` is given.
        prompts: The prompt set. `ENGLISH_PROMPTS` (default) or `POLISH_PROMPTS`,
            or your own `Prompts`. Match it to the document's language.
        max_retries: How many times to retry an LLM call whose output the
            structured parser cannot read, or which the provider rejects with an
            HTTP 400 for a malformed tool call (e.g. Groq `tool_use_failed`). The
            same prompt is re-sent with exponential backoff between attempts; `0`
            disables retrying. Guards against a model that occasionally returns an
            empty or malformed structured answer (some reasoning models do).
        requests_per_minute: When set, the LLM calls (retries included) are spaced
            60 / requests_per_minute seconds apart to stay under a provider's RPM
            limit (e.g. a free tier's 30 RPM). None (default) sends them as fast as
            they arise. Does not throttle the embedder, which runs locally.
        max_concurrency: How many candidate review chains may run at once. The
            default 1 reviews candidates sequentially. Raise it when the endpoint
            handles concurrent requests well (a vLLM/TGI deployment batching
            continuously, most paid APIs): each candidate's classify/verify/merge
            chain then runs on a thread pool this size, and the result is
            identical to a sequential run. Requires an `llm` whose bound
            runnables tolerate concurrent `invoke` calls (LangChain chat models
            do). Composes with `requests_per_minute`, which still caps the total
            rate across all threads.
        knn_k: Neighbours kept per unit when rebuilding the graph. Must match the
            `knn_k` the graph was tiled with.
        primary_k: How many top-similarity units count as direct hits per change.
        similarity_floor: Units below this cosine similarity to the change are
            never sent to the LLM, even when reached via the ripple.

    Returns:
        DetectionResult: one `ChangeImpact` per change, with flat `.relations`
        and `.suggestions` accessors across all changes.

    Note:
        With the default embedder the library owns it and frees its GPU memory
        before the LLM pass. With a custom `embedder`, its lifecycle is yours.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")

    changes = [Change.model_validate(c) for c in changes]

    if not changes:
        return DetectionResult(changes=[])

    units = _flatten_units(tiling)
    owns_embedder = embedder is None

    if owns_embedder:
        kwargs = {"device": device, "dtype": dtype, "batch_size": batch_size}

        if model_name is not None:
            kwargs["model_name"] = model_name

        embedder = SentenceTransformerEmbedder(**kwargs)

    try:
        unit_embeddings = embedder.encode([u.text for u in units], normalize_embeddings=True)
        change_embeddings = embedder.encode([c.text for c in changes], normalize_embeddings=True)
        adjacency = knn_sparsify(unit_embeddings @ unit_embeddings.T, knn_k)
        graph = nx.from_numpy_array(adjacency)
        analyses = [
            _analyze(units, unit_embeddings, change_embeddings[j], graph, primary_k, similarity_floor)
            for j in range(len(changes))
        ]
    finally:
        if owns_embedder:
            embedder.close()

    reviewer = Reviewer(
        llm or default_llm(llm_model),
        prompts,
        max_retries=max_retries,
        requests_per_minute=requests_per_minute,
    )

    if max_concurrency == 1:
        impacts = [_review(change, analysis, reviewer) for change, analysis in zip(changes, analyses)]
    else:
        impacts = _review_concurrent(changes, analyses, reviewer, max_concurrency)

    return DetectionResult(changes=impacts)


@dataclass(frozen=True)
class _Unit:
    id: int
    community: int
    section: str
    text: str


@dataclass(frozen=True)
class _Analysis:
    primary: list[int]
    ripple: list[int]
    candidates: list[_Unit]


def _flatten_units(tiling: TilingResult) -> list[_Unit]:
    units = [
        _Unit(id=u.id, community=c.id, section=u.section, text=" ".join([*u.sentences, *u.payload]))
        for c in tiling.communities
        for u in c.units
    ]

    units.sort(key=lambda u: u.id)

    if [u.id for u in units] != list(range(len(units))):
        raise ValueError("unit ids must be a contiguous 0..n-1 range, as produced by tile()")

    return units


def _analyze(
    units: list[_Unit],
    unit_embeddings: np.ndarray,
    change_embedding: np.ndarray,
    graph: nx.Graph,
    primary_k: int,
    similarity_floor: float,
) -> _Analysis:
    req_sim = unit_embeddings @ change_embedding
    order = np.argsort(-req_sim)
    primary = [int(i) for i in order[:primary_k]]
    ripple = sorted({int(n) for p in primary for n in graph.neighbors(p)} - set(primary))
    impacted_order = sorted(primary, key=lambda i: -req_sim[i]) + sorted(ripple, key=lambda i: -req_sim[i])
    candidates = [units[i] for i in impacted_order if req_sim[i] >= similarity_floor]

    return _Analysis(
        primary=[units[i].id for i in primary],
        ripple=[units[i].id for i in ripple],
        candidates=candidates,
    )


def _review(change: Change, analysis: _Analysis, reviewer: Reviewer) -> ChangeImpact:
    results = [_review_candidate(change, unit, reviewer) for unit in analysis.candidates]

    return _assemble_impact(change, analysis, results)


def _review_concurrent(
    changes: list[Change],
    analyses: list[_Analysis],
    reviewer: Reviewer,
    max_concurrency: int,
) -> list[ChangeImpact]:
    """Runs every (change, candidate) review chain on a shared thread pool.

    Each chain is independent, so chains fan out across changes and within one
    change alike. Results are reassembled in the original candidate order, so
    the output matches the sequential path exactly. On the first failed chain
    no further chains start, in-flight ones finish, and the error propagates.
    """
    results: dict[tuple[int, int], tuple[Relation, Suggestion | None]] = {}

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_review_candidate, change, unit, reviewer): (change_index, candidate_index)
            for change_index, (change, analysis) in enumerate(zip(changes, analyses))
            for candidate_index, unit in enumerate(analysis.candidates)
        }

        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)

            raise

    return [
        _assemble_impact(
            change,
            analysis,
            [results[(change_index, candidate_index)] for candidate_index in range(len(analysis.candidates))],
        )
        for change_index, (change, analysis) in enumerate(zip(changes, analyses))
    ]


def _review_candidate(
    change: Change, unit: _Unit, reviewer: Reviewer
) -> tuple[Relation, Suggestion | None]:
    rel = reviewer.classify(change.text, unit.text)
    verified: bool | None = None
    verify_reason = ""
    suggestion: Suggestion | None = None

    if rel.relation == "strong":
        verdict = reviewer.verify(change.text, unit.text, rel.justification)
        verified = verdict.agrees
        verify_reason = f"[objection] {verdict.objection} [decision] {verdict.reason}"

        if verdict.agrees:
            merged = reviewer.merge(change.text, unit.text)
            suggestion = Suggestion(
                requirement=change.name,
                unit_id=unit.id,
                section=unit.section,
                justification=rel.justification,
                verify_reason=verdict.reason,
                current_text=unit.text,
                added=merged.added,
                merged_text=merged.merged_text,
            )

    relation = Relation(
        requirement=change.name,
        unit_id=unit.id,
        section=unit.section,
        relation=rel.relation,
        unit_topic=rel.unit_topic,
        justification=rel.justification,
        verified=verified,
        verify_reason=verify_reason,
    )

    return relation, suggestion


def _assemble_impact(
    change: Change,
    analysis: _Analysis,
    results: list[tuple[Relation, Suggestion | None]],
) -> ChangeImpact:
    return ChangeImpact(
        name=change.name,
        text=change.text,
        primary=analysis.primary,
        ripple=analysis.ripple,
        relations=[relation for relation, _ in results],
        suggestions=[suggestion for _, suggestion in results if suggestion is not None],
    )
