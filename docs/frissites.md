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

### Átállás a repónkénti mappákra (dwyt-gpu 0.2, dwyt-airouter 0.2)

A 2-es letöltési lista minden fájlt a `gguf/<szervezet>/<repó>/` mappába tesz (a régi 1-es lista a `gguf/`-be). Előbb ezt a modelleszközt frissítsd (lent), utána az AI-routert. A régi helyen lévő fájlokat (közvetlenül a `gguf/` alatt) a letöltő nem használja: ha ugyanabból a repóból valók, áthelyezheted őket (`mkdir -p gguf/unsloth/REPÓ && mv gguf/FÁJL.gguf gguf/unsloth/REPÓ/`), egyébként töröld őket. A `dwyt-gpu check --dir MODELLMAPPA --list downloads.json` megmutatja, mi hiányzik.

## A modelleszköz

```bash
uv tool install --force git+https://github.com/kzkz22/dwyt-gpu
export PATH="/root/.local/bin:$PATH"   # ha az uv szerint a /root/.local/bin nincs a PATH-on (Unraidon minden újraindítás után)
dwyt-gpu --version
```

## Visszaállás

- Worker: `git checkout <előző-commit>`, majd `docker compose build whisper && docker compose create --force-recreate whisper`.
- llama-server: rögzített régebbi kép a `compose.yaml`-ban (lásd fent).
