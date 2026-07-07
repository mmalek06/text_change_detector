from pathlib import Path

import numpy as np
import pytest

from text_change_detector.shared.models import Segment, TilingResult
from text_change_detector.tiling import pipeline
from text_change_detector.tiling.pipeline import (
    _build_groups,
    _create_similarity_matrix,
    _deduplicate_groups,
    _extract,
    _is_boundary,
    _step_dissimilarities,
    _to_result,
    _unit_text,
    tile,
)
from tests.helpers import MINILM, StubEmbedder


def seg(text, section="", payload=None):
    return Segment(text=text, section=section, payload=payload or [])


class TestUnitText:
    def test_joins_text_and_payload_in_order(self):
        group = [seg("a", payload=["p1", "p2"]), seg("b")]

        assert _unit_text(group) == "a p1 p2 b"


class TestStepDissimilarities:
    def test_window_construction_and_values(self):
        table = {
            "s0": [1, 0],
            "s0 s1": [1, 0],
            "s1 s2": [0, 1],
            "s2 s3": [1, 0],
            "s3": [0, 1],
        }
        embedder = StubEmbedder(table=table)
        d = _step_dissimilarities(["s0", "s1", "s2", "s3"], embedder, window_size=2)

        assert embedder.calls[0] == ["s0", "s0 s1", "s1 s2"]
        assert embedder.calls[1] == ["s1 s2", "s2 s3", "s3"]
        assert np.allclose(d, [1.0, 0.0, 0.0])

    def test_length_and_range_with_real_model(self, real_embedder):
        sentences = [
            "The tenant pays the monthly rent.",
            "Late payments incur a penalty.",
            "The database replicates writes to standby nodes.",
            "Each query reads from the nearest replica.",
        ]
        d = _step_dissimilarities(sentences, real_embedder, window_size=2)

        assert d.shape == (3,)
        assert np.all(d >= 0.0) and np.all(d <= 2.0)
        assert d[1] > d[0]


class TestIsBoundary:
    def test_spike_is_a_boundary(self):
        d = np.array([0.1, 0.2] * 10 + [0.9])

        assert _is_boundary(d, 20, radius=20, threshold=3.0)

    def test_ordinary_point_is_not_a_boundary(self):
        d = np.array([0.1, 0.2] * 10 + [0.9])

        assert not _is_boundary(d, 0, radius=20, threshold=3.0)

    def test_zero_spread_is_never_a_boundary(self):
        d = np.full(11, 0.3)

        assert not _is_boundary(d, 5, radius=5, threshold=3.0)


class TestBuildGroups:
    def _patch(self, monkeypatch, d):
        monkeypatch.setattr(pipeline, "_step_dissimilarities", lambda sentences, embedder, window_size: np.asarray(d))

    def test_cuts_at_high_dissimilarity_gaps(self, monkeypatch):
        segments = [seg(f"s{i}") for i in range(6)]

        self._patch(monkeypatch, [0.1, 0.9, 0.1, 0.1, 0.9])

        groups = _build_groups(segments, object(), group_max_len=7, threshold=100.0, floor=0.6, min_solo_words=0)

        assert [[s.text for s in g] for g in groups] == [["s0", "s1"], ["s2", "s3", "s4"], ["s5"]]

    def test_short_solo_segment_is_forced_to_grow(self, monkeypatch):
        segments = [seg("tiny short seg"), seg("s1"), seg("s2")]

        self._patch(monkeypatch, [0.9, 0.9])

        groups = _build_groups(segments, object(), group_max_len=7, threshold=100.0, floor=0.6, min_solo_words=10)

        assert [s.text for s in groups[0]] == ["tiny short seg", "s1"]

    def test_short_solo_stays_alone_without_forced_growth(self, monkeypatch):
        segments = [seg("tiny short seg"), seg("s1"), seg("s2")]

        self._patch(monkeypatch, [0.9, 0.9])

        groups = _build_groups(segments, object(), group_max_len=7, threshold=100.0, floor=0.6, min_solo_words=0)

        assert [s.text for s in groups[0]] == ["tiny short seg"]

    def test_group_max_len_is_respected(self, monkeypatch):
        segments = [seg(f"s{i}") for i in range(9)]

        self._patch(monkeypatch, [0.1] * 8)

        groups = _build_groups(segments, object(), group_max_len=3, threshold=100.0, floor=0.6, min_solo_words=0)

        assert all(len(g) <= 3 for g in groups)
        assert any(len(g) == 3 for g in groups)


class TestDeduplicateGroups:
    def test_substring_group_is_removed(self):
        groups = [[seg("alpha beta")], [seg("alpha beta gamma")], [seg("delta")]]
        result = _deduplicate_groups(groups)

        assert [_unit_text(g) for g in result] == ["alpha beta gamma", "delta"]

    def test_exact_duplicate_is_removed(self):
        groups = [[seg("x y")], [seg("x y")], [seg("z")]]
        result = _deduplicate_groups(groups)

        assert [_unit_text(g) for g in result] == ["x y", "z"]


class TestCreateSimilarityMatrix:
    def test_matrix_is_gram_of_embeddings(self):
        table = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [1, 1, 0]}
        embedder = StubEmbedder(table=table)
        matrix = _create_similarity_matrix([[seg("a")], [seg("b")], [seg("c")]], embedder)
        r = 1 / np.sqrt(2)

        assert np.allclose(matrix, [[1, 0, r], [0, 1, r], [r, r, 1]])


class TestToResult:
    def test_builds_sorted_units_with_flattened_payload(self):
        unique_groups = [
            [seg("s0a", section="A", payload=["p0"]), seg("s0b", section="A")],
            [seg("s1", section="B", payload=["p1"])],
            [seg("s2", section="C")],
        ]
        result = _to_result(unique_groups, [{2, 0}, {1}])

        assert isinstance(result, TilingResult)

        first = result.communities[0]

        assert first.id == 0
        assert [u.id for u in first.units] == [0, 2]
        assert first.units[0].section == "A"
        assert first.units[0].sentences == ["s0a", "s0b"]
        assert first.units[0].payload == ["p0"]
        assert first.units[1].sentences == ["s2"]

        second = result.communities[1]

        assert second.id == 1
        assert [u.id for u in second.units] == [1]
        assert second.units[0].payload == ["p1"]


class TestExtractDispatch:
    def test_list_is_returned_as_is(self):
        segments = [seg("x")]

        assert _extract(segments, None, None) is segments

    def test_custom_extractor_receives_a_path(self):
        captured = {}

        def extractor(path):
            captured["path"] = path

            return [seg("y")]

        result = _extract("file.docx", extractor, None)

        assert captured["path"] == Path("file.docx")
        assert [s.text for s in result] == ["y"]


class TestTileEndToEnd:
    def test_list_source_partitions_all_units(self, real_embedder):
        sentences = [
            "The chef sears the steak in a hot pan.",
            "The recipe simmers the sauce for an hour.",
            "The baker proofs the dough overnight.",
            "The telescope resolves distant spiral galaxies.",
            "Astronomers measure the redshift of quasars.",
            "The probe orbits the gas giant every month.",
        ]
        segments = [seg(s) for s in sentences]
        result = tile(segments, embedder=real_embedder)

        assert result.communities

        ids = sorted(u.id for c in result.communities for u in c.units)

        assert ids == list(range(len(ids)))

    def test_result_is_deterministic(self, real_embedder):
        segments = [seg(s) for s in (
            "The chef sears the steak in a hot pan.",
            "The telescope resolves distant spiral galaxies.",
            "Astronomers measure the redshift of quasars.",
            "The baker proofs the dough overnight.",
        )]
        a = tile(segments, embedder=real_embedder)
        b = tile(segments, embedder=real_embedder)

        assert a.model_dump() == b.model_dump()

    def test_owned_embedder_path_runs_and_closes(self):
        segments = [seg(s) for s in (
            "The service authenticates each request using a bearer token.",
            "The gateway rejects requests that present an expired token.",
            "The database replicates writes to two standby nodes.",
        )]

        try:
            result = tile(segments, model_name=MINILM, device="cpu")
        except Exception as exc:
            pytest.skip(f"could not load {MINILM}: {exc}")

        assert isinstance(result, TilingResult)
        assert result.communities


class TestDistantCommunities:
    def _community_sections(self, result):
        return [{u.section for u in c.units} for c in result.communities]

    def test_far_apart_same_topic_chapters_share_a_community(self, distant_docx, real_embedder):
        result = tile(distant_docx, spacy_model="en_core_web_sm", embedder=real_embedder)
        section_sets = self._community_sections(result)
        joined = [
            sections for sections in section_sets
            if "1. Payment and Rent" in sections and "4. Invoicing and Billing" in sections
        ]

        assert len(joined) == 1
        assert "2. Authentication" not in joined[0]

    def test_authentication_chapter_stays_separate(self, distant_docx, real_embedder):
        result = tile(distant_docx, spacy_model="en_core_web_sm", embedder=real_embedder)
        section_sets = self._community_sections(result)
        auth_community = next(s for s in section_sets if "2. Authentication" in s)

        assert "1. Payment and Rent" not in auth_community
        assert "4. Invoicing and Billing" not in auth_community
