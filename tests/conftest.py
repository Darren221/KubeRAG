from collections.abc import Iterator
from pathlib import Path

import pytest
from fpdf import FPDF

from kuberag.config import Settings


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Iterator[Path]:
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "chroma").mkdir(parents=True)
    yield data


@pytest.fixture
def test_settings(tmp_data_dir: Path) -> Settings:
    return Settings(
        OPENAI_API_KEY="sk-test",
        chroma_path=tmp_data_dir / "chroma",
        bm25_path=tmp_data_dir / "bm25.pkl",
        raw_docs_path=tmp_data_dir / "raw",
    )


@pytest.fixture
def pdf_fixture(tmp_path: Path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "Hello PDF World")
    pdf.add_page()
    pdf.cell(0, 10, "This is page two.")
    out = tmp_path / "sample.pdf"
    pdf.output(str(out))
    return out
