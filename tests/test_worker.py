import json
import time

import httpx

AUDIO = b"fLaC" + b"\0" * 2000


def transcribe(worker, query="model=large-v3&language=hu", body=AUDIO, timeout=60):
    lines = []
    with httpx.stream("POST", f"{worker.url}/transcribe?{query}", content=body, timeout=timeout) as resp:
        if resp.status_code != 200:
            resp.read()
            return resp.status_code, resp.json()
        for raw in resp.iter_lines():
            if raw.strip():
                lines.append(json.loads(raw))
    return 200, lines


def test_job_streams_progress_and_a_diarized_result(worker):
    status, lines = transcribe(worker, "model=large-v3&language=hu&num_speakers=2")
    assert status == 200
    assert any(line.get("stage") for line in lines[:-1])
    result = lines[-1]["result"]
    assert result["device"] == "cuda" and result["diarized"] is True and result["language"] == "hu"
    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_01", "SPEAKER_00"]
    assert isinstance(result["segments"][0]["start"], float)  # numpy scalars became JSON numbers
    assert "diarize cuda 2" in worker.calls()
    assert worker.health()["busy"] is False


def test_gpu_failure_finishes_on_cpu_with_a_warning(worker):
    worker.flag("gpu_fail_once")
    _, lines = transcribe(worker)
    result = lines[-1]["result"]
    assert result["device"] == "cpu"
    assert any("CPU-n fejeződött be" in w for w in result["warnings"])


def test_crash_is_isolated_and_the_rerun_resumes_from_the_checkpoint(worker):
    worker.flag("crash_once")
    _, lines = transcribe(worker)
    assert "error" in lines[-1] and "memóriája" in lines[-1]["error"]
    assert worker.health()["ok"] is True
    before = sum(1 for c in worker.calls() if c.startswith("transcribe"))
    _, lines = transcribe(worker)
    assert "result" in lines[-1]
    assert sum(1 for c in worker.calls() if c.startswith("transcribe")) == before  # Whisper not run again


def test_refusals(make_worker):
    w = make_worker(HF_TOKEN="")
    status, body = transcribe(w, "model=tiny")
    assert status == 422 and "Ismeretlen Whisper-modell" in body["detail"]
    status, body = transcribe(w, "model=large-v3")
    assert status == 422 and "Hugging Face-token" in body["detail"]
    status, lines = transcribe(w, "model=large-v3&diarize=false")
    assert status == 200 and lines[-1]["result"]["diarized"] is False
    status, body = transcribe(w, "model=large-v3&diarize=false", body=b"")
    assert status == 422


def test_too_large_audio_is_read_to_the_end_then_refused(make_worker):
    w = make_worker(MAX_AUDIO_MB="1")
    status, body = transcribe(w, body=b"\0" * (5 * 1024 * 1024))
    assert status == 413 and "MAX_AUDIO_MB" in body["detail"]


def test_a_caller_that_goes_away_stops_the_job_and_frees_the_gpu(worker):
    worker.flag("slow_once")
    with httpx.stream("POST", f"{worker.url}/transcribe?model=large-v3", content=AUDIO, timeout=30) as resp:
        for raw in resp.iter_lines():
            if raw.strip():
                break  # first progress line, then hang up
    deadline = time.time() + 15
    while worker.health()["busy"] and time.time() < deadline:
        time.sleep(0.3)
    assert worker.health()["busy"] is False
    _, lines = transcribe(worker)
    assert "result" in lines[-1]
