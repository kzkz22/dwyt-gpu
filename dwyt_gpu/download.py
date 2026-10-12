"""Model download for the GPU host, driven by `downloads.json`.

`dwyt-airouter generate` writes downloads.json from models.yaml (one file describes every model):

  {"version": 2,
   "files":     [{"repo": "unsloth/Some-GGUF", "path": "Some-Q4_K_M.gguf"}],
   "snapshots": [{"repo": "Systran/faster-whisper-large-v3", "gated": false}]}

files     -> <models_dir>/gguf/<owner>/<repo>/<path>   (llama-server; models.ini points to /models/gguf/...)
             Version 1 lists (older dwyt-airouter) put them straight into gguf/<path>; two repos often hold
             files of the same name (mmproj-F16.gguf), which is why version 2 keeps them apart.
snapshots -> <models_dir>/hf/hub           (Hugging Face cache: the Whisper container has HF_HOME=/models/hf)

A split GGUF (`...-00001-of-00003.gguf`) brings all of its parts. Files already present with the
right size are skipped; the free space is checked before anything is downloaded.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

SPLIT = re.compile(r"^(?P<stem>.*)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$")
MARGIN = 2 * 1024 ** 3  # keep this much free after the download
VERSIONS = (1, 2)


def file_folder(data: dict, entry: dict) -> str:
    """Where a listed file goes, relative to the models dir: gguf/ (version 1) or gguf/<owner>/<repo>."""
    return "gguf" if data.get("version") == 1 else f"gguf/{entry['repo']}"


class DownloadError(Exception):
    """A user-facing reason (Hungarian)."""


@dataclass
class Item:
    kind: str           # "file" or "snapshot"
    repo: str
    path: str = ""      # for files
    folder: str = ""    # for files: the folder under the models dir the path is relative to
    size: int = 0       # bytes still to download
    present: bool = False
    gated: bool = False

    def label(self) -> str:
        return f"{self.repo}/{self.path}" if self.kind == "file" else self.repo


def load_list(path: str | Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DownloadError(f"A letöltési lista nem olvasható ({path}): {exc}") from exc
    if not isinstance(data, dict) or data.get("version") not in VERSIONS:
        raise DownloadError("A letöltési lista nem ismert formátumú (version 1 vagy 2 kell); frissítsd a "
                            "dwyt-gpu-t, vagy készítsd újra: dwyt-airouter generate")
    for entry in data.get("files", []) + data.get("snapshots", []):
        repo = str(entry.get("repo", ""))
        if repo.count("/") != 1 or ".." in repo:
            raise DownloadError(f"Hibás repó a listában: {repo!r}")
    for entry in data.get("files", []):
        p = str(entry.get("path", ""))
        if not p or p.startswith("/") or ".." in Path(p).parts:
            raise DownloadError(f"Hibás fájlnév a listában: {p!r}")
    return data


def expand_parts(path: str) -> list[str]:
    m = SPLIT.match(path)
    if not m:
        return [path]
    total = int(m["total"])
    return [f"{m['stem']}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def plan(data: dict, models_dir: Path, token: str | None) -> list[Item]:
    """What is missing and how big it is. Asks the Hugging Face API for sizes."""
    api = huggingface_hub.HfApi(token=token)
    items: list[Item] = []
    by_repo: dict[str, list[str]] = {}
    folders: dict[str, str] = {}
    for entry in data.get("files", []):
        folders[entry["repo"]] = file_folder(data, entry)
        for part in expand_parts(entry["path"]):
            by_repo.setdefault(entry["repo"], []).append(part)
    for repo, paths in by_repo.items():
        try:
            infos = {i.path: i for i in api.get_paths_info(repo, paths)}
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
            raise DownloadError(f"A {repo} repó nem érhető el: {_reason(exc)}") from exc
        for p in paths:
            info = infos.get(p)
            if info is None:
                raise DownloadError(f"A {repo} repóban nincs ilyen fájl: {p}")
            size = int(getattr(info, "size", 0) or 0)
            local = models_dir / folders[repo] / p
            present = local.is_file() and (not size or local.stat().st_size == size)
            items.append(Item("file", repo, p, folders[repo], 0 if present else size, present))
    for entry in data.get("snapshots", []):
        repo, gated = entry["repo"], bool(entry.get("gated"))
        try:
            huggingface_hub.snapshot_download(repo, cache_dir=str(models_dir / "hf" / "hub"), local_files_only=True)
            items.append(Item("snapshot", repo, present=True, gated=gated))
            continue
        except (LocalEntryNotFoundError, FileNotFoundError, OSError, ValueError):
            pass
        if gated and not token:
            raise DownloadError(f"A {repo} letöltéséhez Hugging Face-token kell (HF_TOKEN), és a modell oldalán el "
                                "kell fogadni a feltételeit.")
        try:
            info = api.model_info(repo, files_metadata=True)
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
            raise DownloadError(f"A {repo} repó nem érhető el: {_reason(exc)}") from exc
        size = sum(int(getattr(s, "size", 0) or 0) for s in (info.siblings or []))
        items.append(Item("snapshot", repo, size=size, gated=gated))
    return items


def _reason(exc: Exception) -> str:
    if isinstance(exc, GatedRepoError):
        return "korlátozott hozzáférésű; fogadd el a feltételeit a modell oldalán, és adj meg HF_TOKEN-t"
    if isinstance(exc, RepositoryNotFoundError):
        return "nem létezik, vagy a token nem fér hozzá"
    return str(exc)[:300]


def human(size: int) -> str:
    return f"{size / 1024 ** 3:.1f} GB" if size >= 1024 ** 3 else f"{size / 1024 ** 2:.0f} MB"


def check_space(items: list[Item], models_dir: Path) -> tuple[int, int]:
    need = sum(i.size for i in items if not i.present)
    free = shutil.disk_usage(models_dir).free
    if need and need + MARGIN > free:
        raise DownloadError(f"Nincs elég hely: {human(need)} kell (+{human(MARGIN)} tartalék), "
                            f"de csak {human(free)} szabad itt: {models_dir}")
    return need, free


def download(items: list[Item], models_dir: Path, token: str | None, say=print) -> None:
    for item in items:
        if item.present:
            say(f"  megvan:   {item.label()}")
            continue
        say(f"  letöltés: {item.label()} ({human(item.size)})")
        try:
            if item.kind == "file":
                huggingface_hub.hf_hub_download(item.repo, item.path, local_dir=str(models_dir / item.folder),
                                                token=token)
            else:
                huggingface_hub.snapshot_download(item.repo, cache_dir=str(models_dir / "hf" / "hub"), token=token)
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError, OSError) as exc:
            raise DownloadError(f"A letöltés megszakadt ({item.label()}): {_reason(exc)}. "
                                "Újraindítva folytatódik.") from exc


def run(list_path: str, models_dir: str, dry_run: bool = False, say=print) -> int:
    data = load_list(list_path)
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    items = plan(data, root, token)
    need, free = check_space(items, root)
    missing = [i for i in items if not i.present]
    say(f"{len(items)} tétel, ebből hiányzik {len(missing)} ({human(need)}); szabad hely: {human(free)}")
    if dry_run:
        for item in items:
            say(f"  {'megvan' if item.present else 'hiányzik'}: {item.label()}"
                + ("" if item.present else f" ({human(item.size)})"))
        return 0
    download(items, root, token, say)
    say("Kész.")
    return 0
