import networkx as nx
import numpy as np
import pytest

from text_change_detector.detection import detect_changes
from text_change_detector.detection.models import (
    Change,
    DetectionResult,
    Merge,
    UnitRelation,
    Verdict,
)
from text_change_detector.detection import pipeline
from text_change_detector.detection.pipeline import _analyze, _flatten_units, _review, _Analysis, _Unit
from text_change_detector.shared.models import Community, SemanticUnit, TilingResult
from tests.helpers import StructuredLLMStub, StubEmbedder


def unit(id, section="", sentences=None, payload=None):
    return SemanticUnit(id=id, section=section, sentences=sentences or [f"s{id}"], payload=payload or [])


def tiling_of(*units, community_id=0):
    return TilingResult(communities=[Community(id=community_id, units=list(units))])


class TestFlattenUnits:
    def test_joins_sentences_and_payload_and_sorts_by_id(self):
        tiling = TilingResult(communities=[
            Community(id=0, units=[unit(2, "C", ["c"], ["pc"])]),
            Community(id=1, units=[unit(0, "A", ["a1", "a2"], ["pa"]), unit(1, "B", ["b"])]),
        ])
        units = _flatten_units(tiling)

        assert [u.id for u in units] == [0, 1, 2]
        assert units[0].text == "a1 a2 pa"
        assert units[0].community == 1
        assert units[2].text == "c pc"

    def test_non_contiguous_ids_raise(self):
        tiling = tiling_of(unit(0), unit(2))

        with pytest.raises(ValueError, match="contiguous"):
            _flatten_units(tiling)


class TestAnalyze:
    def _units(self):
        return [_Unit(id=i, community=0, section="", text=f"u{i}") for i in range(4)]

    def _graph(self):
        graph = nx.Graph()

        graph.add_nodes_from(range(4))
        graph.add_edges_from([(0, 2), (1, 3)])

        return graph

    def test_primary_ripple_and_floor(self):
        units = self._units()
        embeddings = np.array([[0.9], [0.8], [0.4], [0.1]])
        change = np.array([1.0])
        analysis = _analyze(units, embeddings, change, self._graph(), primary_k=2, similarity_floor=0.5)

        assert analysis.primary == [0, 1]
        assert analysis.ripple == [2, 3]
        assert [u.id for u in analysis.candidates] == [0, 1]

    def test_ripple_kept_when_floor_is_low(self):
        units = self._units()
        embeddings = np.array([[0.9], [0.8], [0.4], [0.1]])
        change = np.array([1.0])
        analysis = _analyze(units, embeddings, change, self._graph(), primary_k=2, similarity_floor=0.0)

        assert [u.id for u in analysis.candidates] == [0, 1, 2, 3]


class FakeReviewer:
    def __init__(self, relation_by_text, agrees=True):
        self.relation_by_text = relation_by_text
        self.agrees = agrees
        self.verify_calls = 0
        self.merge_calls = 0

    def classify(self, change, unit):
        return UnitRelation(unit_topic="topic", relation=self.relation_by_text[unit], justification="just")

    def verify(self, change, unit, justification):
        self.verify_calls += 1

        return Verdict(objection="obj", reason="rsn", agrees=self.agrees)

    def merge(self, change, unit):
        self.merge_calls += 1

        return Merge(added="ADD", merged_text="MERGED")


class TestReview:
    def _analysis(self, texts):
        units = [_Unit(id=i, community=0, section=f"S{i}", text=t) for i, t in enumerate(texts)]

        return _Analysis(primary=[0], ripple=[1], candidates=units)

    def test_medium_hit_is_not_verified_or_merged(self):
        analysis = self._analysis(["strong one", "medium one"])
        reviewer = FakeReviewer({"strong one": "strong", "medium one": "medium"})
        impact = _review(Change(name="c", text="chg"), analysis, reviewer)
        medium = next(r for r in impact.relations if r.unit_id == 1)

        assert medium.verified is None
        assert medium.verify_reason == ""
        assert reviewer.verify_calls == 1
        assert [s.unit_id for s in impact.suggestions] == [0]

    def test_strong_verified_produces_suggestion(self):
        analysis = self._analysis(["strong one", "medium one"])
        reviewer = FakeReviewer({"strong one": "strong", "medium one": "medium"}, agrees=True)
        impact = _review(Change(name="c", text="chg"), analysis, reviewer)
        strong = next(r for r in impact.relations if r.unit_id == 0)
        suggestion = impact.suggestions[0]

        assert strong.verified is True
        assert strong.verify_reason == "[objection] obj [decision] rsn"
        assert reviewer.merge_calls == 1
        assert suggestion.added == "ADD"
        assert suggestion.merged_text == "MERGED"
        assert suggestion.current_text == "strong one"
        assert suggestion.verify_reason == "rsn"

    def test_strong_rejected_yields_no_suggestion(self):
        analysis = self._analysis(["strong one"])
        reviewer = FakeReviewer({"strong one": "strong"}, agrees=False)
        impact = _review(Change(name="c", text="chg"), analysis, reviewer)

        assert impact.relations[0].verified is False
        assert impact.suggestions == []
        assert reviewer.merge_calls == 0

    def test_primary_and_ripple_carried_through(self):
        analysis = self._analysis(["strong one", "medium one"])
        reviewer = FakeReviewer({"strong one": "medium", "medium one": "medium"})
        impact = _review(Change(name="c", text="chg"), analysis, reviewer)

        assert impact.primary == [0]
        assert impact.ripple == [1]


def _handlers(relation="strong", agrees=True):
    return {
        UnitRelation: lambda p: UnitRelation(unit_topic="t", relation=relation, justification="j"),
        Verdict: lambda p: Verdict(objection="o", reason="r", agrees=agrees),
        Merge: lambda p: Merge(added="A", merged_text="M"),
    }


class TestDetectChangesEndToEnd:
    def _tiling(self):
        return tiling_of(
            unit(0, "Login", ["login flow"]),
            unit(1, "Labels", ["label printing"]),
            unit(2, "Rent", ["unrelated rent"]),
        )

    def _embedder(self):
        table = {
            "login flow": [1, 0, 0],
            "label printing": [0, 1, 0],
            "unrelated rent": [0, 0, 1],
            "login precondition": [1, 0, 0],
        }

        return StubEmbedder(table=table, dim=3)

    def test_returns_detection_result_with_expected_shape(self):
        llm = StructuredLLMStub(_handlers())
        result = detect_changes(
            self._tiling(),
            [{"name": "login-precondition", "text": "login precondition"}],
            embedder=self._embedder(),
            llm=llm,
            knn_k=1,
            primary_k=1,
            similarity_floor=0.5,
        )

        assert isinstance(result, DetectionResult)

        impact = result.changes[0]

        assert impact.primary == [0]
        assert [r.unit_id for r in impact.relations] == [0]
        assert impact.relations[0].relation == "strong"
        assert impact.relations[0].verified is True
        assert impact.relations[0].verify_reason == "[objection] o [decision] r"
        assert [s.unit_id for s in impact.suggestions] == [0]
        assert result.suggestions[0].merged_text == "M"

    def test_custom_embedder_is_not_closed(self):
        embedder = self._embedder()

        detect_changes(
            self._tiling(),
            [{"name": "c", "text": "login precondition"}],
            embedder=embedder,
            llm=StructuredLLMStub(_handlers()),
            knn_k=1,
            primary_k=1,
        )

        assert not hasattr(embedder, "closed")

    def test_empty_changes_short_circuits(self):
        result = detect_changes(self._tiling(), [])

        assert result.changes == []

    def test_owned_embedder_is_built_and_closed(self, monkeypatch):
        class ClosableStub(StubEmbedder):
            def __init__(self, **kwargs):
                super().__init__(table={
                    "login flow": [1, 0, 0],
                    "label printing": [0, 1, 0],
                    "unrelated rent": [0, 0, 1],
                    "login precondition": [1, 0, 0],
                }, dim=3)
                self.closed = False

            def close(self):
                self.closed = True

        built = ClosableStub()

        monkeypatch.setattr(pipeline, "SentenceTransformerEmbedder", lambda **kwargs: built)
        detect_changes(
            self._tiling(),
            [{"name": "c", "text": "login precondition"}],
            llm=StructuredLLMStub(_handlers()),
            knn_k=1,
            primary_k=1,
        )

        assert built.closed

    def test_default_llm_is_used_when_llm_is_none(self, monkeypatch):
        seen = {}

        def fake_default_llm(model):
            seen["model"] = model

            return StructuredLLMStub(_handlers())

        monkeypatch.setattr(pipeline, "default_llm", fake_default_llm)
        detect_changes(
            self._tiling(),
            [{"name": "c", "text": "login precondition"}],
            embedder=self._embedder(),
            llm_model="my-model:latest",
            knn_k=1,
            primary_k=1,
        )

        assert seen["model"] == "my-model:latest"

    def test_change_objects_and_dicts_are_both_accepted(self):
        result = detect_changes(
            self._tiling(),
            [Change(name="c", text="login precondition")],
            embedder=self._embedder(),
            llm=StructuredLLMStub(_handlers()),
            knn_k=1,
            primary_k=1,
        )

        assert result.changes[0].name == "c"
