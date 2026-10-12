"""dwyt-gpu check: the model folder part (the GPU and Docker parts are faked away)."""
import json

from dwyt_gpu import check, download


def _no_gpu_tools(monkeypatch):
    monkeypatch.setattr(check, "_run", lambda cmd: (0, "NVIDIA Test, 24576 MiB, 1.0"))
    monkeypatch.setattr(check.shutil, "which", lambda name: None)


def test_without_a_folder_it_says_what_was_skipped(monkeypatch):
    _no_gpu_tools(monkeypatch)
    lines = []
    check.run(None, None, say=lines.append)
    assert any(line.startswith("kihagyva") and "Hugging Face-token" in line and "--dir" in line for line in lines)


def test_folder_token_and_the_list_next_to_the_models(tmp_path, monkeypatch):
    _no_gpu_tools(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "models.ini").write_text("version = 1\n")
    download.save_token(tmp_path, "hf_secret")
    (tmp_path / "downloads.json").write_text(json.dumps({"version": 2, "files": [
        {"repo": "org/A-GGUF", "path": "a.gguf"}, {"repo": "org/B-GGUF", "path": "mmproj-F16.gguf"}], "snapshots": []}))
    (tmp_path / "gguf" / "org" / "A-GGUF").mkdir(parents=True)
    (tmp_path / "gguf" / "org" / "A-GGUF" / "a.gguf").write_bytes(b"x")
    lines = []
    check.run(str(tmp_path), None, say=lines.append)
    text = "\n".join(lines)
    assert f"Hugging Face-token: {tmp_path / 'hf' / 'token'}" in text and "hf_secret" not in text
    assert "rendben  GGUF: org/A-GGUF/a.gguf" in text and "HIBA     GGUF: org/B-GGUF/mmproj-F16.gguf" in text
