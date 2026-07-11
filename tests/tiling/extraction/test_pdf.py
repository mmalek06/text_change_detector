from text_change_detector.tiling.extraction.pdf import (
    Block,
    blocks_to_segments,
    body_font_size,
    heading_level,
    join_wrapped,
    running_furniture,
)


def block(text, size=11.0, bold=False, single_line=True, page=0):
    return Block(text=text, size=size, bold=bold, single_line=single_line, page=page)


class TestJoinWrapped:
    def test_joins_with_spaces(self):
        assert join_wrapped(["Hello", "world"]) == "Hello world"

    def test_dehyphenates(self):
        assert join_wrapped(["exam-", "ple"]) == "example"

    def test_mixed_hyphen_and_space(self):
        assert join_wrapped(["a-", "b", "c"]) == "ab c"

    def test_empty(self):
        assert join_wrapped([]) == ""

    def test_single(self):
        assert join_wrapped(["only"]) == "only"


class TestBodyFontSize:
    def test_weights_by_text_length(self):
        blocks = [block("x" * 100, size=11.0), block("y" * 10, size=16.0)]
        assert body_font_size(blocks) == 11.0

    def test_none_when_empty(self):
        assert body_font_size([]) is None


class TestRunningFurniture:
    def test_repeated_header_is_furniture(self):
        bodies = ["alpha body text", "beta body text", "gamma body text", "delta body text"]
        blocks = []

        for page in range(4):
            blocks.append(block(f"Confidential Report Page {page + 1}", size=9.0, page=page))
            blocks.append(block(bodies[page], page=page))

        furniture = running_furniture(blocks, 4)

        assert "Confidential Report Page #" in furniture
        assert all(body not in furniture for body in bodies)

    def test_threshold_is_half_of_pages(self):
        blocks = [block("Shared", page=0), block("Shared", page=1), block("Lonely", page=0)]
        furniture = running_furniture(blocks, 4)

        assert "Shared" in furniture
        assert "Lonely" not in furniture


class TestHeadingLevel:
    def test_numbered_level_one(self, nlp):
        assert heading_level(block("1. Introduction", size=11.0), 11.0, nlp) == 1

    def test_numbered_depth(self, nlp):
        assert heading_level(block("1.2 Access Control", size=11.0), 11.0, nlp) == 2

    def test_all_caps(self, nlp):
        assert heading_level(block("OVERVIEW", size=11.0), 11.0, nlp) == 1

    def test_larger_font(self, nlp):
        assert heading_level(block("Introduction", size=16.0), 11.0, nlp) == 1

    def test_bold_is_level_two(self, nlp):
        assert heading_level(block("Advanced Configuration", size=11.0, bold=True), 11.0, nlp) == 2

    def test_content_is_not_heading(self, nlp):
        assert heading_level(block("The service authenticates each request", size=11.0), 11.0, nlp) is None

    def test_terminal_punctuation_is_not_heading(self, nlp):
        assert heading_level(block("Introduction:", size=16.0), 11.0, nlp) is None

    def test_multi_line_is_not_heading(self, nlp):
        assert heading_level(block("Section Title", size=16.0, single_line=False), 11.0, nlp) is None

    def test_too_long_is_not_heading(self, nlp):
        assert heading_level(block(" ".join(["word"] * 13), size=16.0), 11.0, nlp) is None


def report_blocks():
    pages = [
        ("1. Introduction", 0, [
            "The service authenticates each request using a bearer token.",
            "The gateway rejects requests that present an expired token.",
        ]),
        ("2. Storage", 1, [
            "The database replicates writes to two standby nodes.",
            "__Advanced Configuration__",
            "The engine compacts old segments during nightly maintenance.",
        ]),
        ("3. Networking", 2, [
            "The load balancer distributes traffic across three regions.",
            "Each node reports its health every ten seconds.",
        ]),
        ("4. Billing", 3, [
            "Every invoice becomes due within fourteen days of issuance.",
            "The finance team charges a surcharge for overdue balances.",
        ]),
    ]
    blocks = []

    for heading, page, body in pages:
        blocks.append(block(f"Confidential Report Page {page + 1}", size=9.0, page=page))
        blocks.append(block(heading, size=16.0, page=page))

        for text in body:
            if text.startswith("__"):
                blocks.append(block(text.strip("_"), size=11.0, bold=True, page=page))
            else:
                blocks.append(block(text, size=11.0, page=page))

    return blocks


class TestBlocksToSegments:
    def test_sections_and_running_header_removed(self, nlp):
        segments = blocks_to_segments(report_blocks(), nlp)
        sections = {s.section for s in segments}

        assert {
            "1. Introduction",
            "2. Storage",
            "2. Storage > Advanced Configuration",
            "3. Networking",
            "4. Billing",
        } <= sections
        assert all("Confidential Report" not in s.text for s in segments)

    def test_body_sentences_present(self, nlp):
        extracted = [s.text for s in blocks_to_segments(report_blocks(), nlp)]

        assert "The database replicates writes to two standby nodes." in extracted
        assert "Every invoice becomes due within fourteen days of issuance." in extracted
