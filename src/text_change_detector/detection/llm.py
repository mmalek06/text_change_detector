import threading
import time
from typing import Protocol, runtime_checkable

from langchain_core.exceptions import OutputParserException

from text_change_detector.detection.models import Merge, UnitRelation, Verdict
from text_change_detector.detection.prompts import Prompts

DEFAULT_LLM_MODEL = "gpt-oss:20b"
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 0.5
RETRY_BACKOFF_CAP = 30.0


class StructuredRunnable(Protocol):
    def invoke(self, input: str) -> object: ...


@runtime_checkable
class ChatModel(Protocol):
    """The slice of the LangChain chat-model interface the reviewer relies on.

    Any object with `with_structured_output(schema)` returning a runnable whose
    `invoke(prompt: str)` yields an instance of `schema` works — e.g. a
    `langchain_ollama.ChatOllama`, or any other LangChain chat model.
    """

    def with_structured_output(self, schema: type, **kwargs: object) -> StructuredRunnable: ...


def default_llm(model: str = DEFAULT_LLM_MODEL) -> ChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0)


class Reviewer:
    """Runs the find / verify / merge passes against a single chat model.

    Binds the model to each response schema once (via `with_structured_output`)
    and formats the supplied prompts per call.

    Some models occasionally return output the structured parser cannot read (a
    reasoning model may emit an empty final answer, for example) or make the
    provider reject the request with HTTP 400 (Groq answers a malformed tool call
    with `tool_use_failed`). Both are transient: the same prompt often succeeds on
    a later try. Such a call is retried on the same prompt up to `max_retries`
    times (`0` disables retrying), with exponential backoff between attempts,
    before the failure propagates.

    When `requests_per_minute` is given, every call (retries included) passes
    through a rate limiter that spaces calls to stay under that ceiling, so a free
    tier's RPM limit is not tripped. `None` sends calls as fast as they arise.
    """

    def __init__(
        self,
        llm: ChatModel,
        prompts: Prompts,
        max_retries: int = DEFAULT_MAX_RETRIES,
        requests_per_minute: int | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self._prompts = prompts
        self._max_retries = max_retries
        self._limiter = _RateLimiter(requests_per_minute) if requests_per_minute is not None else None
        self._relation_llm = llm.with_structured_output(UnitRelation)
        self._verify_llm = llm.with_structured_output(Verdict)
        self._merge_llm = llm.with_structured_output(Merge)

    def classify(self, change: str, unit: str) -> UnitRelation:
        return self._invoke(self._relation_llm, self._prompts.relation.format(change=change, unit=unit))

    def verify(self, change: str, unit: str, justification: str) -> Verdict:
        return self._invoke(
            self._verify_llm,
            self._prompts.verify.format(change=change, unit=unit, justification=justification),
        )

    def merge(self, change: str, unit: str) -> Merge:
        return self._invoke(self._merge_llm, self._prompts.merge.format(change=change, unit=unit))

    def _invoke(self, runnable: StructuredRunnable, prompt: str) -> object:
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if self._limiter is not None:
                self._limiter.acquire()

            try:
                result = runnable.invoke(prompt)
            except OutputParserException as exc:
                last_error = exc
            except Exception as exc:
                if not _is_bad_request(exc):
                    raise

                last_error = exc
            else:
                if result is not None:
                    return result

                last_error = OutputParserException("structured output was empty")

            if attempt < self._max_retries:
                time.sleep(_backoff_delay(attempt))

        raise last_error


def _is_bad_request(exc: Exception) -> bool:
    """Whether `exc` is a provider HTTP 400 (e.g. Groq `tool_use_failed`).

    Detected by shape, not type, so the library depends on no LLM SDK: the
    OpenAI / Groq / Anthropic client errors carry `status_code`, an httpx error
    carries `response.status_code`.
    """
    if getattr(exc, "status_code", None) == 400:
        return True

    response = getattr(exc, "response", None)

    return getattr(response, "status_code", None) == 400


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff, in seconds, for a zero-based failed-attempt index."""
    return min(RETRY_BACKOFF_CAP, RETRY_BACKOFF_BASE * (2**attempt))


class _RateLimiter:
    """Spaces synchronous calls to honour a requests-per-minute ceiling.

    Reserves evenly spaced slots `60 / requests_per_minute` seconds apart and
    sleeps until the reserved slot is due, so a burst is stretched to the allowed
    rate instead of being rejected. Thread-safe, so one reviewer can be shared.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")

        self._min_interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._min_interval
            wait = slot - now

        if wait > 0:
            time.sleep(wait)
