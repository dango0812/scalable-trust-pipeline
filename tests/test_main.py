import runpy
import sys
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "dango0812-github-nickname.txt"


def test_python_dash_m_runs_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "trustpipe",
        "--lake",
        str(tmp_path),
        "ingest",
        str(SAMPLE),
        "--source",
        "github.com",
        "--source-uri",
        "https://github.com/dango0812",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("trustpipe", run_name="__main__")
    assert exit_info.value.code == 0
    assert '"stage": "bronze.ingest"' in capsys.readouterr().out
