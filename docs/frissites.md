# A GPU-gép frissítése

A konténereket az AI-router indítja. Frissítés előtt nézd meg, fut-e épp munka (`docker ps`): futó konténert ne hozz létre újra.

## A Whisper-worker

```bash
cd /opt/dwyt-gpu
git pull
docker compose build whisper
docker compose create --force-recreate whisper
```

A kész átírási lépések (`/data`) és a modellek (`/models`) megmaradnak. Unraidon: `git pull` a forrásmappában, `docker build -t dwyt-whisper:local .../worker`, majd a sablonban **Apply**, ami újra létrehozza a konténert. Utána állítsd le.

## A llama-server

```bash
docker compose pull llama
docker compose create --force-recreate llama
```

Unraidon a Docker fülön **Check for Updates**, utána **Apply**, majd leállítás. A llama.cpp gyorsan változik. Ha egy új változat nem indul vagy hibázik, a `compose.yaml`-ban rögzíts egy korábbi képet (`ghcr.io/ggml-org/llama.cpp:server-cuda-bNNNN`), és hozd létre újra.

## Új vagy módosított modell

1. Az AI-routeren: `models.yaml` szerkesztése, majd `dwyt-airouter generate`.
2. Ide: a `models.ini` a modellmappába, és `dwyt-gpu download downloads.json --dir MODELLMAPPA`.
3. A llama-server a következő indításkor már az új presetekkel indul. Ha éppen fut, állítsd le (`docker stop dwyt-llama`), és a következő kérés újraindítja.

Régi GGUF-fájlt a letöltő nem töröl. Ha már nem kell, töröld kézzel a `gguf/` mappából.

## A modelleszköz

```bash
uv tool upgrade dwyt-gpu
```

## Visszaállás

- Worker: `git checkout <előző-commit>`, majd `docker compose build whisper && docker compose create --force-recreate whisper`.
- llama-server: rögzített régebbi kép a `compose.yaml`-ban (lásd fent).
