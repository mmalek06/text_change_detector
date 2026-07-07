# text-change-detector

Split text into semantic units and detect which units a proposed change
impacts, using embeddings, graph clustering and a local LLM.

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
embedding model and `knn_k`** as tiling — the defaults already match, so it just
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
`Prompts` (each template is a `str.format` string — `relation` and `merge` use
`{change}` / `{unit}`, `verify` also uses `{justification}`):

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
returning — and, in `detect_changes`, before the LLM pass runs, so the embedder
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
