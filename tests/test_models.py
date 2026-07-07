from text_change_detector.shared.models import Community, SemanticUnit, TilingResult


class TestTilingResultTree:
    def test_nested_structure_and_serialisation(self):
        result = TilingResult(
            communities=[
                Community(
                    id=0,
                    units=[SemanticUnit(id=0, section="s", sentences=["hi"], payload=[])],
                )
            ]
        )

        assert result.communities[0].units[0].sentences == ["hi"]
        assert result.model_dump() == {
            "communities": [
                {"id": 0, "units": [{"id": 0, "section": "s", "sentences": ["hi"], "payload": []}]}
            ]
        }

    def test_round_trips_through_json(self):
        result = TilingResult(
            communities=[Community(id=1, units=[SemanticUnit(id=9, section="x", sentences=["a"], payload=["n"])])]
        )

        assert TilingResult.model_validate_json(result.model_dump_json()) == result

    def test_empty_result(self):
        assert TilingResult(communities=[]).communities == []
