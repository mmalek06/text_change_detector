# text-change-detector

Split text into semantic units and detect which units a proposed change
impacts, using embeddings, graph clustering and a local LLM.

## How it compares

The segmentation step is a modern take on **topic segmentation**, a problem with
a long pre-embedding history. Its reference point is Hearst's **TextTiling**
(1997): score the dissimilarity between adjacent blocks of text at every
candidate gap, then place a boundary wherever that dissimilarity spikes.
TextTiling (and relatives such as C99 or LCseg) measured cohesion *lexically*:
bag-of-words vectors, word overlap, lexical chains. That makes them blind to
paraphrase and synonymy; two sentences about the same topic in different words
score as unrelated.

This project keeps the TextTiling skeleton and swaps its signal. The
dissimilarity at each gap is the cosine distance between the **sentence
embeddings** of the window to its left and right (`_step_dissimilarities`), so
cohesion is judged by meaning rather than shared tokens. A gap becomes a boundary
either when similarity drops below a hard `floor`, or when it stands out as a
**robust z-score** (median + MAD) against its *local* neighbourhood
(`_is_boundary`). This is an adaptive, per-region test rather than one global
threshold, so a locally sharp topic shift is caught even inside a dense passage,
while a single noisy sentence in a calm one is not mistaken for a boundary.
Around each seed the window grows in both directions up to `group_max_len`, and
very short
solo fragments are held back (`min_solo_words`) instead of standing alone. A
second stage then lifts the linear segmentation into structure: the units are
embedded, connected into a kNN similarity graph, and clustered with **Louvain**
community detection, so thematically related units are grouped into communities
even when they are not adjacent in the document.

### Cuts along topics, not to a size

This matters because the goal here differs from the splitters shipped out of the
box with RAG toolkits. The structural ones (`RecursiveCharacterTextSplitter` and
friends) cut on separators to hit a **target chunk size** with overlap; the
embedding-aware ones (LangChain's `SemanticChunker`, LlamaIndex's
`SemanticSplitterNodeParser`) improve on that by cutting where the distance
between consecutive sentence embeddings crosses a percentile threshold, but they
still walk the text greedily, left to right, optimising for retrieval-friendly
chunks. This tiler is built to cut along **thematic boundaries**: a window-based
dissimilarity judged against local robust statistics, plus the graph-community
layer, so a unit is a coherent topic span rather than a size-bounded slice. If
you need fixed-size chunks for a vector store, reach for those tools; if you need
to know *where the document actually changes subject* (which is what
change-impact analysis rests on), that is what this does.

### Generic by design

The defaults are meant to be left alone. The algorithm is generic: the boundary
test is relative (a robust z-score against the local baseline) and the signal is
a normalized embedding cosine, so nothing is calibrated to a particular document
length, domain or writing style, and the same settings are meant to carry from a
short PDF spec to a long Word requirements document. The knobs on `tile()`
(`window_size`, `baseline_radius`, `threshold`, `floor`, `group_max_len`,
`min_solo_words`, `knn_k`) are exposed as an escape hatch for unusual inputs, not
as dials you are expected to sweep. Reach for them only when a specific document
segments in a way you want to nudge.

## Installation

```bash
uv add text-change-detector
```

### spaCy models

The built-in extractors need a spaCy model for the document's language. The
models are not bundled (PyPI forbids URL dependencies), so add the ones you need
to your project with uv:

```bash
uv add "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
uv add "pl_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/pl_core_news_sm-3.8.0/pl_core_news_sm-3.8.0-py3-none-any.whl"
```

Adding them with `uv add` (instead of `python -m spacy download`) records them in
your lockfile, so `uv sync` will not prune them.

## Usage

The library works in two steps. `tile` splits a document into semantic units on a
kNN relation graph; `detect_changes` takes that graph plus a list of proposed
changes and reports which units each change impacts, then drafts a merged text
for the ones an LLM confirms.

### 1. Tiling

```python
from text_change_detector import tile

tiling = tile("spec.docx", spacy_model="en_core_web_sm")

print(tiling.model_dump_json(indent=2))
```

`tile` accepts a path, an already-loaded python-docx / PyMuPDF document, or a
`list[Segment]` you built yourself, and returns a `TilingResult`. `spacy_model`
is required only when a built-in extractor runs (skip it if you pass `extractor=`
or a `list[Segment]`). Pass `embedder=` to use your own embedding model (any
object with `encode(list[str], normalize_embeddings=True) -> np.ndarray`);
otherwise a default SentenceTransformer is used, configurable via `model_name`,
`device`, `dtype`, `batch_size`.

### 2. Detection (tiling + detection end to end)

```python
from text_change_detector import tile, detect_changes, Change

tiling = tile("spec.docx", spacy_model="en_core_web_sm")
changes = [
    Change(
        name="two-factor-at-checkout",
        text=(
            "Customers paying by card must confirm the purchase with a one-time code "
            "sent to their phone. The order is not placed until the code is verified."
        ),
    ),
]
result = detect_changes(tiling, changes)

for suggestion in result.suggestions:
    print(f"[{suggestion.requirement}] unit #{suggestion.unit_id} ({suggestion.section})")
    print("  before:", suggestion.current_text)
    print("  after :", suggestion.merged_text)
```

`changes` is always yours to supply; each item is a `Change` (or a dict with
`name` and `text`). Detection rebuilds the graph, so it must use the **same
embedding model and `knn_k`** as tiling; the defaults already match, so it just
works unless you override them.

The default LLM runs the passes on a local [Ollama](https://ollama.com) server,
so pull the model first (or point `llm_model=` at one you have):

```bash
ollama pull gpt-oss:20b
```

### Detection in another language

Keep the document, the changes and the prompt set in the same language. Polish
prompts ship with the library:

```python
from text_change_detector import tile, detect_changes, Change, POLISH_PROMPTS

tiling = tile("specyfikacja.pdf", spacy_model="pl_core_news_sm")
result = detect_changes(
    tiling,
    [Change(
        name="limit-wypozyczen",
        text=(
            "Czytelnik może wypożyczyć jednocześnie najwyżej pięć książek. "
            "Po osiągnięciu limitu system blokuje kolejne wypożyczenie do czasu zwrotu."
        ),
    )],
    prompts=POLISH_PROMPTS,
)
```

### Bring your own LLM and prompts

`llm=` takes any LangChain chat model that supports
`with_structured_output(schema).invoke(prompt)`; `prompts=` takes your own
`Prompts` (each template is a `str.format` string, where `relation` and `merge`
use `{change}` / `{unit}` and `verify` also uses `{justification}`):

```python
from langchain_ollama import ChatOllama
from text_change_detector import detect_changes, Change, Prompts

my_llm = ChatOllama(model="qwen3:30b-a3b-instruct", temperature=0)
my_prompts = Prompts(
    relation="Rate how the change relates to the unit.\nCHANGE:\n{change}\n\nUNIT:\n{unit}",
    verify="Does the change belong here?\nCHANGE:\n{change}\n\nUNIT:\n{unit}\n\nWHY: {justification}",
    merge="Apply the change, editing as little as possible.\nCHANGE:\n{change}\n\nUNIT:\n{unit}",
)
result = detect_changes(tiling, changes, llm=my_llm, prompts=my_prompts)
```

### Reading the result

`detect_changes` returns a `DetectionResult`: one `ChangeImpact` per change, plus
flat accessors across all changes.

```python
result.changes        # list[ChangeImpact], one per change
result.relations      # flat: every reviewed unit/change relation
result.suggestions    # flat: verified-strong edits carrying current_text -> merged_text

impact = result.changes[0]
impact.primary        # unit ids the change directly resembles (direct hits)
impact.ripple         # unit ids one graph hop away (the wider review set)
impact.relations      # per candidate: relation, justification, verified, verify_reason
impact.suggestions    # the confirmed edits for this change
```

### GPU memory

With the default embedder the library owns it and frees its GPU memory before
returning, and in `detect_changes` before the LLM pass runs, so the embedder
and a local LLM can share one GPU. With a custom `embedder` the library never
touches its lifecycle, so releasing GPU memory (and not starving the LLM of it)
is your responsibility.

## Known limitations

- **TODO (docx footnotes):** In its current form the `.docx` extractor does not
  take Word footnotes into account - footnote text lives in a separate document
  part (`word/footnotes.xml`) and is not read, so any "side notes" it carries do
  not become part of a semantic unit or community. This is planned to change in a
  future version, where footnotes will be pulled in and attached to the segment
  that references them.
- **TODO (units embedded twice):** Running the full pipeline embeds the semantic
  units twice - `tile` embeds each unit to build the relation graph, then
  `detect_changes` embeds the same units again to rebuild it, because a
  `TilingResult` carries only text, not the vectors or the graph. On a large
  document that is a redundant pass over the embedding model. A future version
  could let `tile` hand the unit embeddings (or the graph) to `detect_changes` so
  they are computed once, leaving only the change texts to embed.

## Status

Early alpha. API is not yet stable.
