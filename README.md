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

```python
from text_change_detector import tile

result = tile("spec.docx", spacy_model="en_core_web_sm")

print(result.model_dump_json(indent=2))
```

`tile` accepts a path, an already-loaded python-docx / PyMuPDF document, or a
`list[Segment]` you built yourself, and returns a `TilingResult`. `spacy_model`
is required only when a built-in extractor runs (skip it if you pass `extractor=`
or a `list[Segment]`). Pass `embedder=` to use your own embedding model (any
object with `encode(list[str], normalize_embeddings=True) -> np.ndarray`);
otherwise a default SentenceTransformer is used, configurable via `model_name`,
`device`, `dtype`, `batch_size`.

GPU memory: with the default embedder the library owns it and frees its GPU
memory before returning. With a custom `embedder` the library never touches its
lifecycle, so releasing GPU memory is your responsibility.

## Known limitations

- **TODO (docx footnotes):** In its current form the `.docx` extractor does not
  take Word footnotes into account - footnote text lives in a separate document
  part (`word/footnotes.xml`) and is not read, so any "side notes" it carries do
  not become part of a semantic unit or community. This is planned to change in a
  future version, where footnotes will be pulled in and attached to the segment
  that references them.

## Status

Early alpha. API is not yet stable.
