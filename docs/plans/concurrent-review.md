# Plan: concurrent execution path for the LLM review stage

## Goal

Add a `max_concurrency` option to `detect_changes` so the LLM review stage can keep
several requests in flight at once. This is aimed at self-hosted OpenAI-compatible
endpoints (e.g. vLLM on RunPod serverless) where continuous batching multiplies
aggregate throughput, so a run that takes many GPU-hours sequentially finishes in a
fraction of the wall-clock time and cost. The sequential path stays untouched and
remains the default.

## Design decisions

1. **Threads, not asyncio.** The `ChatModel` protocol
   (`with_structured_output(...).invoke(prompt)`) is synchronous and every existing
   adapter satisfies it. An async path would require a second protocol (`ainvoke`),
   a duplicated `Reviewer` and a duplicated pipeline for no gain: the work is pure
   I/O, so the GIL does not matter. LangChain chat models tolerate concurrent
   `invoke`; the protocol docstring documents that requirement.
2. **Unit of parallelism: the (change, candidate) chain.** For each candidate the
   `classify -> (verify -> merge)` sequence is an independent chain. Fanning chains
   out parallelizes both across changes and within a single change. Results are
   assembled in the original candidate order per change and input order across
   changes, so the output is identical to the sequential path.
3. **Error semantics unchanged:** full result or an exception. On the first failed
   chain the pool stops accepting work (`shutdown(cancel_futures=True)`), in-flight
   chains finish, and the first error propagates.
4. **`requests_per_minute` still applies globally.** The rate limiter is already
   thread-safe, so both knobs compose: RPM caps the total rate, `max_concurrency`
   caps the number of requests in flight.

## Changes

- `src/text_change_detector/detection/pipeline.py`
  - `detect_changes(..., max_concurrency: int = 1)` with validation (`>= 1`).
  - Extract the per-candidate body of `_review` into
    `_review_candidate(change, unit, reviewer) -> (Relation, Suggestion | None)`.
  - New `_review_concurrent(...)`: one `ThreadPoolExecutor(max_workers=max_concurrency)`
    for the whole run, one task per (change, candidate) chain, ordered assembly,
    fail-fast on the first error.
  - Dispatch: `max_concurrency == 1` runs today's loop, `> 1` the pool.
- `src/text_change_detector/detection/llm.py`: docstrings only (concurrent-`invoke`
  requirement on `ChatModel`, thread-safety note on `Reviewer`).
- `README.md`: document `max_concurrency` with an OpenAI-compatible endpoint example.
- `pyproject.toml`: version 0.6.0 -> 0.7.0.

## Tests

- New stub in `tests/helpers.py`: a `StructuredLLMStub` variant whose every call
  waits on a shared `threading.Barrier`, proving genuine overlap deterministically
  (a barrier timeout fails the test instead of hanging it).
- New `TestDetectChangesConcurrent` in `tests/detection/test_pipeline.py`:
  - concurrent result equals the sequential result on the same stub,
  - calls genuinely overlap under `max_concurrency > 1` (barrier stub),
  - an exception in one chain propagates and does not hang the run,
  - `max_concurrency=0` is rejected with `ValueError`,
  - the shared rate limiter sees every LLM call on the concurrent path,
  - the pool size matches `max_concurrency`.

## Out of scope

- Retry-after / HTTP 429 handling (no rate limits on a self-hosted endpoint).
- Partial results on mid-run failure.
- Consumer-side changes (bumping the git dependency, pointing the notebook at the
  new endpoint) happen in the consuming project after this lands.

## Definition of done

1. `uv run pytest` green, with the pre-existing tests unmodified.
2. `uv run ruff check` green.
3. `detect_changes` with `max_concurrency=4` returns the same result as with the
   default on the same inputs.
