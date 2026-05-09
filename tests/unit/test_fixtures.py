from pathlib import Path

from kuberag.config import Settings


def test_tmp_data_dir_creates_subdirs(tmp_data_dir: Path) -> None:
    assert tmp_data_dir.is_dir()
    assert (tmp_data_dir / "raw").is_dir()
    assert (tmp_data_dir / "chroma").is_dir()


def test_test_settings_uses_tmp_paths(test_settings: Settings, tmp_data_dir: Path) -> None:
    assert test_settings.chroma_path == tmp_data_dir / "chroma"
    assert test_settings.bm25_path == tmp_data_dir / "bm25.pkl"
    assert test_settings.raw_docs_path == tmp_data_dir / "raw"
    assert test_settings.openai_api_key.get_secret_value() == "sk-test"
