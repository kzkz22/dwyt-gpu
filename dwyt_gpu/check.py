"""Checks of the GPU host: NVIDIA driver, Docker with GPU access, the model folder."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from . import download


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc.returncode, (proc.stdout or proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def run(models_dir: str | None, list_path: str | None, say=print) -> int:
    ok = True

    def report(good: bool, text: str) -> None:
        nonlocal ok
        ok &= good
        say(f"{'rendben' if good else 'HIBA   '}  {text}")

    rc, out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    report(rc == 0, f"NVIDIA GPU: {out.splitlines()[0] if rc == 0 and out else 'nem található (nvidia-smi)'}")
    if not shutil.which("docker"):
        report(False, "Docker: nincs telepítve")
    else:
        rc, out = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
        runtimes = json.loads(out) if rc == 0 and out.startswith("{") else {}
        rc_cdi, cdi = _run(["docker", "info", "--format", "{{json .CDISpecDirs}}"])
        gpu = "nvidia" in runtimes or (rc_cdi == 0 and "cdi" in cdi.lower())
        report(rc == 0, "Docker: fut" if rc == 0 else f"Docker: nem érhető el ({out[:200]})")
        report(gpu, "Docker GPU-hozzáférés: " + ("nvidia runtime" if "nvidia" in runtimes else
                                                 "CDI" if gpu else "nincs (NVIDIA Container Toolkit kell)"))
        for name in ("dwyt-llama", "dwyt-whisper"):
            rc, out = _run(["docker", "inspect", "--format", "{{.State.Status}}", name])
            report(rc == 0, f"konténer {name}: {out if rc == 0 else 'nincs létrehozva (docker compose create)'}")
    if models_dir:
        root = Path(models_dir)
        report(root.is_dir(), f"modellmappa: {root}")
        if root.is_dir():
            report((root / "models.ini").is_file(), "models.ini a modellmappában"
                   + ("" if (root / "models.ini").is_file() else " (dwyt-airouter generate készíti; másold ide)"))
            say(f"         szabad hely: {download.human(shutil.disk_usage(root).free)}")
            if list_path:
                data = download.load_list(list_path)
                for entry in data.get("files", []):
                    folder = download.file_folder(data, entry)
                    for part in download.expand_parts(entry["path"]):
                        report((root / folder / part).is_file(), f"GGUF: {folder.removeprefix('gguf/')}/{part}"
                               if folder != "gguf" else f"GGUF: {part}")
    say("Minden rendben." if ok else "Van javítanivaló (lásd fent).")
    return 0 if ok else 1
