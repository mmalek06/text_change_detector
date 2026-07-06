import pytest
import spacy

from text_change_detector.embedder import SentenceTransformerEmbedder
from tests import helpers


@pytest.fixture(scope="session")
def nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model 'en_core_web_sm' is not installed")


@pytest.fixture(scope="session")
def real_embedder():
    try:
        embedder = SentenceTransformerEmbedder()
    except Exception as exc:
        pytest.skip(f"could not load the default embedding model: {exc}")

    yield embedder

    embedder.close()


@pytest.fixture
def legal_docx(tmp_path):
    return helpers.build_legal_docx(tmp_path / "legal.docx")


@pytest.fixture
def it_docx(tmp_path):
    return helpers.build_it_docx(tmp_path / "it.docx")


@pytest.fixture
def toc_docx(tmp_path):
    return helpers.build_toc_docx(tmp_path / "toc.docx")


@pytest.fixture
def table_docx(tmp_path):
    return helpers.build_table_docx(tmp_path / "tables.docx")


@pytest.fixture
def footnote_docx(tmp_path):
    return helpers.build_footnote_docx(tmp_path / "footnote.docx")


@pytest.fixture
def distant_docx(tmp_path):
    return helpers.build_distant_topics_docx(tmp_path / "distant.docx")


@pytest.fixture
def report_pdf(tmp_path):
    return helpers.build_report_pdf(tmp_path / "report.pdf")
