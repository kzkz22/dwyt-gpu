# dwyt-gpu

The **dwyt** GPU host package (NVIDIA, Linux + Docker; Windows via WSL2): llama-server container, the Whisper
GPU worker (port of the old yt-transcribe gpu-worker: WhisperX large-v3/turbo → wav2vec2 alignment → pyannote
diarization, NDJSON streaming with heartbeats, CUDA failure → CPU retry), model cache on NVMe, and a
model-download command driven by the airouter's `models.yaml`. Started and stopped on demand by dwyt-airouter;
its ports must only be reachable from the AI router (no auth of its own).

Status: planning only. License: MIT.

## Conventions

- The owner writes in Hungarian. User-facing docs (`README.md`, `docs/`) and any UI are Hungarian;
  code, comments and this file are English.
- The plan and the decision log live in `dwyt-docs` (`docs/terv.md`, section 20). Read the relevant
  sections before changing anything; when a decision changes, update it there in the same piece of work.
- The central–runner protocol is specified in `dwyt-docs/docs/protokoll.md`; both sides follow it.
- Never commit tokens or keys (API keys, OAuth secrets, runner/MCP tokens). Secrets live in env files or
  the runner's local store, never in the repo.
- Add a test for every behaviour change.
- The old system (`yt-transcribe`, `yt-transcribe-webui`) stays separate and unchanged; code may be
  ported from it, but nothing here may depend on it at runtime.
