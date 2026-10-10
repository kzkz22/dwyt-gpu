# dwyt-gpu

A **dwyt** GPU-gépe: NVIDIA GPU-s gépre (Linux + Docker, Unraid; Windowson WSL2-vel) a llama-server és a Whisper GPU-worker konténere, a modellek mappája és egy modellletöltő parancs. A konténereket az AI-router ([dwyt-airouter](https://github.com/kzkz22/dwyt-airouter)) indítja és állítja le igény szerint, egyszerre mindig csak egyet. Így a GPU-n az LLM és a Whisper felváltva fut.

- **`dwyt-llama`:** a llama.cpp hivatalos `server-cuda` képe router módban. A presetek a `models.ini`-ből jönnek, amit a `dwyt-airouter generate` készít.
- **`dwyt-whisper`:** WhisperX (large-v3 / large-v3-turbo) → wav2vec2-igazítás → pyannote-beszélőfelismerés. Egy munka egyetlen HTTP-kérés, a haladást NDJSON-sorokban küldi. CUDA-hiba után CPU-n fejezi be, a Whisper-lépés eredményét megőrzi.
- **`dwyt-gpu download`:** letölti a `downloads.json` szerinti modelleket a Hugging Face-ről, előtte ellenőrzi a helyet. **`dwyt-gpu check`:** meghajtó, Docker GPU-hozzáféréssel, konténerek, modellmappa.

Telepítés: [docs/telepites.md](docs/telepites.md) · frissítés: [docs/frissites.md](docs/frissites.md) · terv: [dwyt-docs, 10. fejezet](https://github.com/kzkz22/dwyt-docs/blob/main/docs/terv.md)

A konténereknek nincs saját belépésük: a portjaikat tűzfallal csak az AI-router érje el.

## Fejlesztés

```bash
uv venv && uv pip install -e ".[test]"
uv run pytest -q
```

A tesztek a valódi workert futtatják alfolyamatként, a torch és a whisperx helyett utánzattal (`tests/fakes`). A Hugging Face is utánzat. GPU nem kell.

## Licenc

MIT, lásd [LICENSE](LICENSE).
