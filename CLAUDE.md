# dwyt-gpu

The **dwyt** GPU host package (NVIDIA, Linux + Docker; Windows via WSL2): llama-server container, the Whisper
GPU worker (port of the old yt-transcribe gpu-worker: WhisperX large-v3/turbo → wav2vec2 alignment → pyannote
diarization, NDJSON streaming with heartbeats, CUDA failure → CPU retry), model cache on NVMe, and a
model-download command driven by the airouter's `models.yaml`. Started and stopped on demand by dwyt-airouter;
its ports must only be reachable from the AI router (no auth of its own).

Status: round 2 implemented (worker port, compose file, `dwyt-gpu download|check`); not yet run on a real GPU host.
License: MIT.

## Layout

| Path | Role |
| --- | --- |
| `worker/gpuworker/server.py` | Worker HTTP API: one job per streamed request (NDJSON + 10 s heartbeats), one job at a time |
| `worker/gpuworker/job.py` | Child process per job; CUDA failure -> CPU retry with a warning |
| `worker/gpuworker/asr.py` | WhisperX -> alignment -> diarization; ASR checkpoint keyed by audio SHA-256 + model + language |
| `worker/Dockerfile` | `whisperx==3.8.6` pinned (signatures verified against it in the old system) |
| `compose.yaml` | `dwyt-llama` (llama.cpp server-cuda, router mode, `/models/models.ini`) and `dwyt-whisper`; create only |
| `dwyt_gpu/download.py` | `downloads.json` (written by `dwyt-airouter generate`) -> `<dir>/gguf/<owner>/<repo>/` (version 2; version 1: flat `<dir>/gguf`), `<dir>/hf/hub`; space check; HF token from `HF_TOKEN` or `<dir>/hf/token` (0600, `dwyt-gpu token`) |
| `dwyt_gpu/check.py` | Driver, Docker GPU access, containers, model folder |

Worker messages that reach the user are Hungarian. Tests run the real worker with `tests/fakes` (torch,
whisperx) first on PYTHONPATH; flag files switch fake behaviour (crash_once, gpu_fail_once, slow_once).

## Conventions

- The owner writes in Hungarian. User-facing docs (`README.md`, `docs/`) and any UI are Hungarian;
  code, comments and this file are English.
- The plan and the decision log live in `dwyt-docs` (`docs/terv.md`, section 21). Read the relevant
  sections before changing anything; when a decision changes, update it there in the same piece of work.
- The central–runner protocol is specified in `dwyt-docs/docs/protokoll.md`; both sides follow it.
- Never commit tokens or keys (API keys, OAuth secrets, runner/MCP tokens). Secrets live in env files or
  the runner's local store, never in the repo.
- Add a test for every behaviour change.
- The old system (`yt-transcribe`, `yt-transcribe-webui`) stays separate and unchanged; code may be
  ported from it, but nothing here may depend on it at runtime.
