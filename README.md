# text-change-detector

Split text into semantic units and detect which units a proposed change
impacts, using embeddings, graph clustering and a local LLM.

## Why this exists

This is not an attempt to add another brick to the general RAG stack. It targets
one concrete job: **carrying a change through an existing document** when a new
requirement or idea arrives and has to be reflected in the specification, the
contract, or whatever text is being analyzed. The question it answers
is "given this new rule, which parts of the document does it touch, and how
should each of them now read?"

For that job, splitting the work into a deterministic embedding-and-graph stage
and a narrow LLM stage is more stable and cheaper than handing the whole document
and the change to an end-to-end LLM. Retrieval runs once and is reproducible; the
LLM is then asked only about a short list of candidate units, one change against
one unit at a time, and never has to hold the entire document in context or
rewrite it wholesale. That bounds the token cost, keeps every edit local and
auditable, and leaves far less room for the drift and hallucination you invite
when a single long generation is responsible for a whole document.

It runs fully locally out of the box (a local SentenceTransformer embedder and a
local Ollama model), so nothing has to leave the machine. But local is the floor,
not the ceiling: the LLM and the embedder are both injectable, so you can point
the verify and merge passes at a more powerful model (any LangChain chat model,
hosted frontier models included) or swap in a stronger embedder for the
semantic-unit matching, and keep the same pipeline around it.

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
when it is a **peak** of that dissimilarity whose **prominence** stands out from
the surrounding curve (`_prominence_boundaries`), which is the modern form of
Hearst's own depth score. Prominence is a height difference, how far the curve
must descend on either side of a peak before it climbs again, so it does not
depend on the absolute cosine scale of any particular embedder, and it reads only
the nearest valleys rather than a wide statistical window, so it holds up even
where boundaries are dense. The four-sentence window is what makes a boundary
require a sustained change rather than a single sentence: one off-topic sentence
is diluted inside the window and barely moves the signal. Around each seed the
window grows in both directions up to `group_max_len`, and very short solo
fragments are held back (`min_solo_words`) instead of standing alone. A
second stage then lifts the linear segmentation into structure: the units are
embedded, connected into a kNN similarity graph, and clustered with **Louvain**
community detection, so thematically related units are grouped into communities
even when they are not adjacent in the document.

### Why units stay short

Both stages lean on one property of the embedding: a semantic unit is encoded as
a single vector, and the more sentences that vector has to summarise, the more it
drifts toward a blurry average that blends several topics and discriminates
poorly. `group_max_len` caps a unit at seven sentences to keep each vector sharp
and topically focused, so the similarities that drive both boundary detection and
the change-to-unit ranking stay meaningful. The specific value of seven was
chosen empirically, from trying a handful of documents rather than a formal
sweep, which is part of why it stays a tunable parameter.

The usual cost of short units, a single topic scattered across several small
fragments, does not hurt here, because regrouping them is exactly what the
Louvain stage is for. Community detection collects every unit about the same
subject into one topical community, so the pipeline gets sharp per-unit
embeddings and document-level topic grouping at the same time. Keeping units
short is therefore the safe default: shortness is cheap to recover from at the
community stage, whereas an over-long, blurred unit is not.

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
test is the prominence of a peak in a normalized embedding cosine, a height
difference rather than an absolute level, so nothing is calibrated to a particular
document length, domain, writing style, or embedding model, and the same settings
are meant to carry from a short PDF spec to a long Word requirements document. The
knobs on `tile()` (`window_size`, `prominence_c`, `group_max_len`,
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

### PDF reading

The core ships no PDF engine and carries no PDF engine's licence. To tile a PDF
you inject a reading strategy: any `PdfReader`, that is a callable turning a PDF
path into a `list[Block]`. A ready-made one lives in a companion package:

```bash
uv add "text-change-detector-pymupdf-adapter @ git+https://github.com/mmalek06/text_change_detector_pymupdf_adapter"
```

It is backed by PyMuPDF, which is AGPL-3.0, so adding it makes the combined work
AGPL. The core itself stays MIT; if the AGPL does not fit your project, inject a
reader backed by a permissively licensed engine (or your own `pdf_reader=`).
`.docx` tiling needs no reader.

Pass the reader to `tile` via `pdf_reader=`:

```python
from pathlib import Path

from text_change_detector import tile
from text_change_detector_pymupdf_adapter import read_blocks

tiling = tile(Path("spec.pdf"), spacy_model="en_core_web_sm", pdf_reader=read_blocks)
```

A reader is any `PdfReader`: a callable taking a PDF path and returning a
`list[Block]`. The shared `blocks_to_segments` turns those blocks into segments
the same way regardless of engine, so writing your own is just the read step:

```python
from pathlib import Path

from text_change_detector import Block, tile

def read_blocks(source) -> list[Block]:
    # parse `source` with your PDF engine of choice and return one Block per
    # paragraph: Block(text, size, bold, single_line, page)
    ...

tiling = tile(Path("spec.pdf"), spacy_model="en_core_web_sm", pdf_reader=read_blocks)
```

## Usage

The library works in two steps. `tile` splits a document into semantic units on a
kNN relation graph; `detect_changes` takes that graph plus a list of proposed
changes and reports which units each change impacts, then drafts a merged text
for the ones an LLM confirms.

### 1. Tiling

```python
from pathlib import Path

from text_change_detector import tile

tiling = tile(Path("spec.docx"), spacy_model="en_core_web_sm")

print(tiling.model_dump_json(indent=2))
```

`tile` accepts a raw text string, a `pathlib.Path` to a file, an already-loaded
python-docx document, or a `list[Segment]` you built yourself, and returns a
`TilingResult`. Only a `Path` is read from disk; a plain `str` is always treated
as raw text and split into sentences, so wrap file names in `Path(...)`.
`spacy_model` is required when a built-in extractor runs or when the source is a
raw text string (skip it only if you pass `extractor=` or a `list[Segment]`); a
`.pdf` path also needs `pdf_reader=` (see [PDF reading](#pdf-reading)). Pass
`embedder=` to use your own embedding model (any object with
`encode(list[str], normalize_embeddings=True) -> np.ndarray`); otherwise a default
SentenceTransformer is used, configurable via `model_name`, `device`, `dtype`,
`batch_size`.

To tile a plain text string, pass it straight in:

```python
from text_change_detector import tile

tiling = tile(
    "The tenant pays the monthly rent by the fifth day of each month. "
    "Late payments incur a penalty of five percent of the outstanding balance.",
    spacy_model="en_core_web_sm",
)
```

### 2. Detection (tiling + detection end to end)

```python
from pathlib import Path

from text_change_detector import tile, detect_changes, Change

tiling = tile(Path("spec.docx"), spacy_model="en_core_web_sm")
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
from pathlib import Path

from text_change_detector import tile, detect_changes, Change, POLISH_PROMPTS
from text_change_detector_pymupdf_adapter import read_blocks

tiling = tile(Path("specyfikacja.pdf"), spacy_model="pl_core_news_sm", pdf_reader=read_blocks)
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

The shipped prompts are deliberately general. The tiling and graph machinery is
domain-agnostic, and so are `ENGLISH_PROMPTS` and `POLISH_PROMPTS`, which speak
about "a change" and "a unit" in neutral terms. That generality is a floor, not
a ceiling: in a specialised domain the neutral wording can leave quality on the
table, so it is worth teaching the prompts the domain's own rules. In a legal
document, for instance, you might tell the model to weigh only the text currently
in force and to ignore repealed provisions (the `(uchylony)` stubs), to respect
article and paragraph boundaries, or to treat a cross-reference to another
article as context rather than as the unit a change belongs in. Because
`prompts=` is just a `Prompts` of `str.format` templates, this is a copy-and-edit
away, with no change to the library.

### Concurrent reviews against a batching endpoint

By default the review chains (classify, then verify and merge for strong hits)
run one candidate at a time. That is the right pace for a local single-stream
model or a free tier with a strict requests-per-minute cap, but it wastes an
endpoint that batches concurrent requests server-side, such as a vLLM or TGI
deployment or most paid APIs. Against those, raise `max_concurrency` to keep
that many chains in flight at once; the result is identical to a sequential
run, only faster:

```python
from langchain_openai import ChatOpenAI

from text_change_detector import detect_changes


class JsonSchemaLLM:
    """Binds every response schema through the endpoint's json_schema mode,
    which vLLM enforces with guided decoding for any model it serves."""

    def __init__(self, llm):
        self._llm = llm

    def with_structured_output(self, schema, **kwargs):
        return self._llm.with_structured_output(schema, method="json_schema", **kwargs)


llm = JsonSchemaLLM(ChatOpenAI(model="my-model", base_url="https://my-endpoint/v1", api_key="..."))
result = detect_changes(tiling, changes, llm=llm, max_concurrency=8)
```

The `llm` you pass must tolerate concurrent `invoke` calls (LangChain chat
models do). `requests_per_minute` still applies globally when set: it caps the
total rate across all threads, while `max_concurrency` caps how many requests
are in flight.

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

## Taskboard

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
- **TODO (unit length vs embedding size):** `group_max_len` is a fixed 7 for
  every embedder. But how many sentences a single vector can hold before it blurs
  (see "Why units stay short") depends on the embedding's capacity, so a
  higher-dimensional model could safely take longer units while a smaller one may
  want shorter ones. A future version could derive the cap from the embedder's
  dimensionality (or ship a per-model default) instead of using the same constant
  for every model.

## Status

Early alpha. API is not yet stable.
