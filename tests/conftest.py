"""Test harness: the real Whisper worker as a subprocess, with stand-ins for torch and whisperx.

`tests/fakes` goes first on the worker's PYTHONPATH. The fakes record their calls in
`<state>/calls.log` and behave differently when flag files exist in `<state>`:
crash_once (the job process is killed), gpu_fail_once (CUDA error in diarization),
slow_once (transcription takes 40 s).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKES = ROOT / "tests" / "fakes"
WORKER = ROOT / "worker"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Worker:
    def __init__(self, tmp: Path, **env: str) -> None:
        self.tmp, self.state = tmp, tmp / "state"
        self.state.mkdir(parents=True)
        self.port = free_port()
        full = {**os.environ, "FAKE_DIR": str(self.state), "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": f"{FAKES}{os.pathsep}{WORKER}", "DATA_DIR": str(tmp / "data"),
                "HF_TOKEN": "hf_fake", "FAKE_CUDA": "1", **env}
        self.log = open(tmp / "worker.log", "ab")
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "gpuworker.server:app", "--host", "127.0.0.1",
                                      "--port", str(self.port), "--log-level", "warning"],
                                     env={k: v for k, v in full.items() if v is not None}, cwd=WORKER,
                                     stdout=self.log, stderr=self.log)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                self.health()
                return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("the worker did not start")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def health(self) -> dict:
        with urllib.request.urlopen(self.url + "/health", timeout=2) as resp:
            return json.loads(resp.read())

    def flag(self, name: str) -> None:
        (self.state / name).touch()

    def calls(self) -> list[str]:
        f = self.state / "calls.log"
        return f.read_text().splitlines() if f.exists() else []

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


@pytest.fixture
def make_worker(tmp_path):
    made: list[Worker] = []

    def _make(**env: str) -> Worker:
        w = Worker(tmp_path / f"w{len(made)}", **env)
        made.append(w)
        return w

    yield _make
    for w in made:
        w.stop()


@pytest.fixture
def worker(make_worker):
    return make_worker()
