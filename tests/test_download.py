import json
import types

import huggingface_hub
import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from dwyt_gpu import cli, download

MB = 1024 ** 2


class FakeHub:
    """Stands in for the Hugging Face API: two repos with sizes, downloads write small files."""

    def __init__(self, monkeypatch):
        self.files = {"org/Big-GGUF": {"Big-00001-of-00002.gguf": 30 * MB, "Big-00002-of-00002.gguf": 20 * MB},
                      "org/Small-GGUF": {"small.gguf": 5 * MB},
                      "org/A-GGUF": {"a.gguf": 4 * MB, "mmproj-F16.gguf": 1 * MB},
                      "org/B-GGUF": {"b.gguf": 4 * MB, "mmproj-F16.gguf": 2 * MB}}
        self.snapshots = {"Systran/faster-whisper-large-v3": 3 * MB, "pyannote/gated": 1 * MB}
        self.downloaded: list[str] = []
        self.cached: set[str] = set()
        hub = self

        class Api:
            def __init__(self, token=None):
                self.token = token

            def get_paths_info(self, repo, paths):
                return [types.SimpleNamespace(path=p, size=hub.files[repo][p]) for p in paths if p in hub.files[repo]]

            def model_info(self, repo, files_metadata=False):
                return types.SimpleNamespace(siblings=[types.SimpleNamespace(size=hub.snapshots[repo])])

        def hf_hub_download(repo, path, local_dir, token=None):
            target = __import__("pathlib").Path(local_dir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\0" * 10)
            hub.downloaded.append(f"{repo}/{path}")
            return str(target)

        def snapshot_download(repo, cache_dir, token=None, local_files_only=False):
            if local_files_only:
                if repo in hub.cached:
                    return cache_dir
                raise LocalEntryNotFoundError("not cached")
            hub.cached.add(repo)
            hub.downloaded.append(repo)
            return cache_dir

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)


def write_list(tmp_path, gated=False):
    path = tmp_path / "downloads.json"
    path.write_text(json.dumps({"version": 1,
                                "files": [{"repo": "org/Big-GGUF", "path": "Big-00001-of-00002.gguf"},
                                          {"repo": "org/Small-GGUF", "path": "small.gguf"}],
                                "snapshots": [{"repo": "Systran/faster-whisper-large-v3"}]
                                + ([{"repo": "pyannote/gated", "gated": True}] if gated else [])}))
    return path


def test_downloads_all_parts_and_snapshots_then_skips_what_is_there(tmp_path, monkeypatch, capsys):
    hub = FakeHub(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    models = tmp_path / "models"
    assert cli.main(["download", str(write_list(tmp_path)), "--dir", str(models)]) == 0
    assert hub.downloaded == ["org/Big-GGUF/Big-00001-of-00002.gguf", "org/Big-GGUF/Big-00002-of-00002.gguf",
                              "org/Small-GGUF/small.gguf", "Systran/faster-whisper-large-v3"]
    assert (models / "gguf" / "small.gguf").is_file()
    # sizes now match? the fake writes 10 bytes, so make them match the listed size for the rerun
    for name, size in {**hub.files["org/Big-GGUF"], **hub.files["org/Small-GGUF"]}.items():
        (models / "gguf" / name).write_bytes(b"\0" * size)
    hub.downloaded.clear()
    assert cli.main(["download", str(write_list(tmp_path)), "--dir", str(models)]) == 0
    assert hub.downloaded == []
    assert "hiányzik 0" in capsys.readouterr().out


def test_not_enough_space_stops_before_downloading(tmp_path, monkeypatch, capsys):
    hub = FakeHub(monkeypatch)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=1 * 1024 ** 3))
    assert cli.main(["download", str(write_list(tmp_path)), "--dir", str(tmp_path / "m")]) == 1
    assert hub.downloaded == [] and "Nincs elég hely" in capsys.readouterr().err


def test_gated_model_needs_a_token(tmp_path, monkeypatch, capsys):
    FakeHub(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert cli.main(["download", str(write_list(tmp_path, gated=True)), "--dir", str(tmp_path / "m")]) == 1
    assert "Hugging Face-token" in capsys.readouterr().err
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert cli.main(["download", str(write_list(tmp_path, gated=True)), "--dir", str(tmp_path / "m"),
                     "--dry-run"]) == 0


@pytest.mark.parametrize("bad", [{"version": 3},
                                 {"version": 1, "files": [{"repo": "a/b", "path": "../etc/passwd"}]},
                                 {"version": 1, "files": [{"repo": "../x", "path": "a.gguf"}]}])
def test_malformed_lists_are_refused(tmp_path, bad):
    path = tmp_path / "l.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(download.DownloadError):
        download.load_list(path)


def test_version_2_keeps_same_named_files_of_two_repos_apart(tmp_path, monkeypatch, capsys):
    hub = FakeHub(monkeypatch)
    lst = tmp_path / "downloads.json"
    lst.write_text(json.dumps({"version": 2, "files": [
        {"repo": "org/A-GGUF", "path": "a.gguf"}, {"repo": "org/A-GGUF", "path": "mmproj-F16.gguf"},
        {"repo": "org/B-GGUF", "path": "b.gguf"}, {"repo": "org/B-GGUF", "path": "mmproj-F16.gguf"}],
        "snapshots": []}))
    models = tmp_path / "models"
    assert cli.main(["download", str(lst), "--dir", str(models)]) == 0
    assert hub.downloaded == ["org/A-GGUF/a.gguf", "org/A-GGUF/mmproj-F16.gguf",
                              "org/B-GGUF/b.gguf", "org/B-GGUF/mmproj-F16.gguf"]
    for repo in ("A", "B"):
        assert (models / "gguf" / "org" / f"{repo}-GGUF" / "mmproj-F16.gguf").is_file()
    assert not (models / "gguf" / "mmproj-F16.gguf").exists()
    # the second repo's projector is not taken for present because the first one is there
    (models / "gguf" / "org" / "A-GGUF" / "mmproj-F16.gguf").write_bytes(b"\0" * MB)
    items = download.plan(download.load_list(lst), models, None)
    present = {f"{i.repo}/{i.path}" for i in items if i.present}
    assert present == {"org/A-GGUF/mmproj-F16.gguf"}


def test_token_from_the_models_dir_saved_root_only(tmp_path, monkeypatch, capsys):
    import os
    import stat
    hub = FakeHub(monkeypatch)
    seen = []
    monkeypatch.setattr(huggingface_hub, "whoami", lambda token=None: seen.append(token) or {"name": "zoltan"})
    monkeypatch.delenv("HF_TOKEN", raising=False)
    models = tmp_path / "models"
    assert download.find_token(models) == (None, "")
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "  hf_secret  ")
    assert cli.main(["token", "--dir", str(models)]) == 0
    path = models / "hf" / "token"
    assert path.read_text() == "hf_secret\n" and seen == ["hf_secret"]
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    out = capsys.readouterr().out
    assert "hf_secret" not in out and "zoltan" in out
    assert download.find_token(models) == ("hf_secret", str(path))
    monkeypatch.setenv("HF_TOKEN", "hf_env")  # the environment wins
    assert download.find_token(models)[0] == "hf_env"
    monkeypatch.delenv("HF_TOKEN")
    # the download uses the stored token (a gated repo needs one)
    assert cli.main(["download", str(write_list(tmp_path, gated=True)), "--dir", str(models), "--dry-run"]) == 0
    assert "Hugging Face-token: " + str(path) in capsys.readouterr().out
    assert cli.main(["token", "--dir", str(models), "--remove"]) == 0 and not path.exists()
    assert hub is not None


def test_a_refused_token_is_not_saved(tmp_path, monkeypatch, capsys):
    from huggingface_hub.utils import HfHubHTTPError

    def refuse(token=None):
        err = HfHubHTTPError.__new__(HfHubHTTPError)  # its constructor differs between hub versions
        Exception.__init__(err, "401 Unauthorized")
        raise err
    monkeypatch.setattr(huggingface_hub, "whoami", refuse)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "hf_bad")
    assert cli.main(["token", "--dir", str(tmp_path)]) == 1
    assert not (tmp_path / "hf" / "token").exists() and "nem fogadta el" in capsys.readouterr().err
