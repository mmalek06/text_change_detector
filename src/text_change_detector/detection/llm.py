from typing import Protocol, runtime_checkable

from text_change_detector.detection.models import Merge, UnitRelation, Verdict
from text_change_detector.detection.prompts import Prompts

DEFAULT_LLM_MODEL = "gpt-oss:20b"


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
    """

    def __init__(self, llm: ChatModel, prompts: Prompts) -> None:
        self._prompts = prompts
        self._relation_llm = llm.with_structured_output(UnitRelation)
        self._verify_llm = llm.with_structured_output(Verdict)
        self._merge_llm = llm.with_structured_output(Merge)

    def classify(self, change: str, unit: str) -> UnitRelation:
        return self._relation_llm.invoke(self._prompts.relation.format(change=change, unit=unit))

    def verify(self, change: str, unit: str, justification: str) -> Verdict:
        return self._verify_llm.invoke(
            self._prompts.verify.format(change=change, unit=unit, justification=justification)
        )

    def merge(self, change: str, unit: str) -> Merge:
        return self._merge_llm.invoke(self._prompts.merge.format(change=change, unit=unit))
