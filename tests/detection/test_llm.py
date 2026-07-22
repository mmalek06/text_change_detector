from types import SimpleNamespace

import pytest
from langchain_core.exceptions import OutputParserException

from text_change_detector.detection import llm as llm_module
from text_change_detector.detection.llm import (
    DEFAULT_MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RETRY_BACKOFF_CAP,
    Reviewer,
    _backoff_delay,
    _is_bad_request,
)
from text_change_detector.detection.models import UnitRelation
from text_change_detector.detection.prompts import Prompts

PROMPTS = Prompts(
    relation="{change}|{unit}",
    verify="{change}|{unit}|{justification}",
    merge="{change}|{unit}",
)


def relation():
    return UnitRelation(unit_topic="t", relation="none", justification="j")


def bad_request():
    exc = RuntimeError("Error code: 400 - tool_use_failed")
    exc.status_code = 400

    return exc


def bad_request_via_response():
    exc = RuntimeError("Error code: 400")
    exc.response = SimpleNamespace(status_code=400)

    return exc


def server_error():
    exc = RuntimeError("Error code: 500")
    exc.status_code = 500

    return exc


class FlakyRunnable:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


class FlakyLLM:
    def __init__(self, runnable):
        self.runnable = runnable

    def with_structured_output(self, schema, **kwargs):
        return self.runnable


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()

    monkeypatch.setattr(llm_module, "time", fake)

    return fake


class TestIsBadRequest:
    def test_status_code_400_is_a_bad_request(self):
        assert _is_bad_request(bad_request())

    def test_response_status_code_400_is_a_bad_request(self):
        assert _is_bad_request(bad_request_via_response())

    def test_other_status_code_is_not_a_bad_request(self):
        assert not _is_bad_request(server_error())

    def test_plain_exception_is_not_a_bad_request(self):
        assert not _is_bad_request(RuntimeError("boom"))


class TestBackoffDelay:
    def test_grows_exponentially_from_the_base(self):
        assert _backoff_delay(0) == RETRY_BACKOFF_BASE
        assert _backoff_delay(1) == RETRY_BACKOFF_BASE * 2
        assert _backoff_delay(2) == RETRY_BACKOFF_BASE * 4

    def test_is_capped(self):
        assert _backoff_delay(100) == RETRY_BACKOFF_CAP


class TestReviewerRetries:
    def test_bad_request_is_retried_then_succeeds(self, clock):
        good = relation()
        runnable = FlakyRunnable([bad_request(), good])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS)

        assert reviewer.classify("c", "u") == good
        assert runnable.calls == 2

    def test_bad_request_via_response_is_retried(self, clock):
        good = relation()
        runnable = FlakyRunnable([bad_request_via_response(), good])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS)

        assert reviewer.classify("c", "u") == good
        assert runnable.calls == 2

    def test_persistent_bad_request_raises_after_all_attempts(self, clock):
        runnable = FlakyRunnable([bad_request() for _ in range(DEFAULT_MAX_RETRIES + 1)])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS)

        with pytest.raises(RuntimeError):
            reviewer.classify("c", "u")

        assert runnable.calls == DEFAULT_MAX_RETRIES + 1

    def test_non_400_error_is_not_retried(self, clock):
        runnable = FlakyRunnable([server_error(), relation()])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS)

        with pytest.raises(RuntimeError):
            reviewer.classify("c", "u")

        assert runnable.calls == 1

    def test_parse_failure_is_retried(self, clock):
        good = relation()
        runnable = FlakyRunnable([OutputParserException("bad"), good])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS)

        assert reviewer.classify("c", "u") == good
        assert runnable.calls == 2

    def test_empty_result_is_retried(self, clock):
        good = relation()
        runnable = FlakyRunnable([None, good])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS)

        assert reviewer.classify("c", "u") == good
        assert runnable.calls == 2

    def test_zero_retries_makes_one_attempt(self, clock):
        runnable = FlakyRunnable([bad_request(), relation()])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS, max_retries=0)

        with pytest.raises(RuntimeError):
            reviewer.classify("c", "u")

        assert runnable.calls == 1
        assert clock.slept == []

    def test_client_controls_the_retry_count(self, clock):
        good = relation()
        runnable = FlakyRunnable([bad_request() for _ in range(5)] + [good])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS, max_retries=5)

        assert reviewer.classify("c", "u") == good
        assert runnable.calls == 6

    def test_negative_retries_is_rejected(self):
        with pytest.raises(ValueError):
            Reviewer(FlakyLLM(FlakyRunnable([])), PROMPTS, max_retries=-1)


class TestReviewerBackoff:
    def test_retries_wait_with_exponential_backoff(self, clock):
        runnable = FlakyRunnable([bad_request() for _ in range(4)] + [relation()])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS, max_retries=4)

        reviewer.classify("c", "u")

        assert clock.slept == [0.5, 1.0, 2.0, 4.0]

    def test_no_backoff_after_the_final_attempt(self, clock):
        runnable = FlakyRunnable([bad_request(), bad_request()])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS, max_retries=1)

        with pytest.raises(RuntimeError):
            reviewer.classify("c", "u")

        assert clock.slept == [0.5]


class TestReviewerRateLimit:
    def test_no_throttling_by_default(self, clock):
        runnable = FlakyRunnable([relation(), relation()])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS)

        reviewer.classify("c", "u")
        reviewer.classify("c", "u")

        assert clock.slept == []

    def test_calls_are_spaced_to_the_rpm_interval(self, clock):
        runnable = FlakyRunnable([relation(), relation(), relation()])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS, requests_per_minute=30)

        reviewer.classify("c", "u")
        reviewer.classify("c", "u")
        reviewer.classify("c", "u")

        assert clock.slept == [2.0, 2.0]

    def test_retries_are_also_throttled(self, clock):
        runnable = FlakyRunnable([bad_request(), relation()])
        reviewer = Reviewer(FlakyLLM(runnable), PROMPTS, requests_per_minute=30)

        reviewer.classify("c", "u")

        assert runnable.calls == 2
        assert clock.slept == [0.5, 1.5]


class TestRateLimiter:
    def test_first_call_does_not_sleep_then_spaces_by_interval(self, clock):
        limiter = llm_module._RateLimiter(30)

        for _ in range(3):
            limiter.acquire()

        assert clock.slept == [2.0, 2.0]

    def test_elapsed_time_reduces_the_wait(self, clock):
        limiter = llm_module._RateLimiter(60)

        limiter.acquire()
        clock.now += 0.4
        limiter.acquire()

        assert clock.slept == [pytest.approx(0.6)]

    def test_non_positive_rpm_is_rejected(self):
        with pytest.raises(ValueError):
            llm_module._RateLimiter(0)
