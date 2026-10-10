# A GPU-gép telepítése

Ez a leírás a dwyt GPU-gépének első telepítését írja le. Ide kerül a llama-server (helyi LLM-ek) és a Whisper GPU-worker (helyi átírás beszélőkkel) konténere és a modellek. A konténereket nem ez a gép indítja, hanem az AI-router ([dwyt-airouter](https://github.com/kzkz22/dwyt-airouter)), egyszerre mindig csak egyet. A frissítés: [frissites.md](frissites.md). A terv: [dwyt-docs, 10. fejezet](https://github.com/kzkz22/dwyt-docs/blob/main/docs/terv.md).

Két út van:

- **A. Linux + Docker:** Debian, Ubuntu vagy más Linux. Windowson WSL2-ben, de csak ha az AI-router is ugyanott fut.
- **B. Unraid:** sablonból létrehozott konténerekkel.

A nagybetűs részeket (`AI-ROUTER-IP`, `MODELLMAPPA`) cseréld a saját értékeidre.

| Elem | Konténernév | Port | Mire |
| --- | --- | --- | --- |
| llama-server | `dwyt-llama` | 8001 | helyi LLM-ek (router mód, a presetek a `models.ini`-ből) |
| Whisper GPU-worker | `dwyt-whisper` | 8766 | átírás, igazítás, beszélőfelismerés |
| modellmappa | – | – | NVMe-n: `gguf/` (LLM-fájlok), `hf/` (Whisper és társai), `models.ini` |

## Előfeltételek

- NVIDIA GPU és meghajtó: `nvidia-smi` kiírja a kártyát.
- Docker és az NVIDIA Container Toolkit. Próba: `docker run --rm --gpus all ubuntu nvidia-smi` kiírja a kártyát.
- Hugging Face-fiók és olvasási token a beszélőfelismeréshez. A [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) oldalán el kell fogadni a feltételeket.
- Hely a modellmappában (lásd A.1, „A modellmappa”): egy 30B körüli GGUF 18–20 GB, a két Whisper-modell kb. 4,5 GB.

## A. Linux + Docker

### A.1 A program és a konténerek

```bash
# a GPU-gépen
git clone https://github.com/kzkz22/dwyt-gpu /opt/dwyt-gpu
cd /opt/dwyt-gpu
cp whisper.env.example whisper.env && chmod 600 whisper.env
nano whisper.env        # HF_TOKEN=hf_... (a Whisper-modellek listája és a többi maradhat)

export DWYT_MODELS_DIR=/srv/dwyt-models        # gyors lemezen
mkdir -p $DWYT_MODELS_DIR
echo "DWYT_MODELS_DIR=$DWYT_MODELS_DIR" > .env
# ha az AI-router ugyanezen a gépen fut: echo "DWYT_BIND=127.0.0.1" >> .env

docker compose build whisper     # a Whisper-worker képe (pár GB, több perc)
docker compose pull llama
docker compose create            # csak létrehozza, nem indítja
```

A konténerek indítása és leállítása az AI-router dolga, ne legyen automatikus indításuk. A `compose.yaml` ezért `restart: "no"`.

**A modellmappa.** A modellek a gépen a `DWYT_MODELS_DIR` mappába kerülnek. Ez a fenti példában `/srv/dwyt-models`; ha nem adod meg, a `compose.yaml` is ezt használja. Válassz olyan helyet, ahol gyors lemez (NVMe) és elég szabad hely van, például `/mnt/nvme/dwyt-models`. A mappát a `.env` jegyzi meg a `compose.yaml` mellett. Mindkét konténer `/models`-ként látja, ugyanúgy, mint Unraidon:

| A gépen | A konténerben | Mi van benne | Ki tölti fel |
| --- | --- | --- | --- |
| `DWYT_MODELS_DIR/gguf/` | `/models/gguf/` | az LLM-ek GGUF-fájljai | `dwyt-gpu download` |
| `DWYT_MODELS_DIR/hf/` | `/models/hf/` | Whisper, igazító- és beszélőfelismerő modellek (Hugging Face-gyorsítótár) | `dwyt-gpu download`; ami hiányzik, azt a worker első használatkor maga tölti le |
| `DWYT_MODELS_DIR/models.ini` | `/models/models.ini` | a llama-server presetjei | átmásolod az AI-routerről |

A worker saját munkaadatai (a félbemaradt munkák folytatásához) külön Docker-kötetbe kerülnek (`dwyt-gpu_whisper-data`); ez kicsi.

### A.2 A modelleszköz

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # ha még nincs uv
uv tool install git+https://github.com/kzkz22/dwyt-gpu
dwyt-gpu --version
```

### A.3 Hálózat

A konténereknek nincs belépésük, ezért a 8001-es és a 8766-os portot csak az AI-router érje el. Ha az AI-router ugyanez a gép, a `DWYT_BIND=127.0.0.1` elég. Különben tűzfallal, például `ufw`-vel:

```bash
ufw allow from AI-ROUTER-IP to any port 8001,8766 proto tcp
```

Ha a `docker-ssh` indítót használod, az AI-router `airouter` felhasználója SSH-kulccsal lép be ide. Ennek a felhasználónak a `docker` csoportban kell lennie. A kulcs beállítását az AI-router leírása mutatja.

## B. Unraid

A két konténert a Docker fülön, kézzel létrehozott sablonnal készíted el (**Add Container**). Mindkettőnél legyen kikapcsolva az **Autostart**, és a GPU-hoz kell az `--runtime=nvidia` és a `NVIDIA_VISIBLE_DEVICES=all` (a Nvidia Driver bővítmény szerint).

A modellmappa legyen a gyorsítótár-meghajtón, például `/mnt/cache/appdata/dwyt-models`.

**`dwyt-llama`:**

| Mező | Érték |
| --- | --- |
| Repository | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| Extra Parameters | `--runtime=nvidia` |
| Post Arguments | `--host 0.0.0.0 --port 8001 --models-preset /models/models.ini --models-max 1 --jinja --flash-attn on --metrics` |
| Port | 8001 → 8001 |
| Path | `/models` → `/mnt/cache/appdata/dwyt-models` |
| Variable | `NVIDIA_VISIBLE_DEVICES=all` |

**`dwyt-whisper`:** a képet egyszer helyben kell megépíteni (Unraid terminál):

```bash
git clone https://github.com/kzkz22/dwyt-gpu /mnt/cache/appdata/dwyt-gpu-src
docker build -t dwyt-whisper:local /mnt/cache/appdata/dwyt-gpu-src/worker
```

| Mező | Érték |
| --- | --- |
| Repository | `dwyt-whisper:local` |
| Extra Parameters | `--runtime=nvidia` |
| Port | 8766 → 8766 |
| Path | `/models` → `/mnt/cache/appdata/dwyt-models` |
| Path | `/data` → `/mnt/cache/appdata/dwyt-whisper-data` |
| Variable | `NVIDIA_VISIBLE_DEVICES=all`, `HF_TOKEN=hf_...` |

Létrehozás után az Unraid elindítja a konténert. Állítsd le, ezentúl az AI-router indítja.

A modelleszköz az Unraid terminálján is fut. Az uv magával hozza a Pythont, de az Unraid újraindításkor elfelejti (a gyökér-fájlrendszer a memóriában van), ezért letöltés előtt újra kell telepíteni:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source /root/.local/bin/env
uv tool install git+https://github.com/kzkz22/dwyt-gpu
export DWYT_MODELS_DIR=/mnt/cache/appdata/dwyt-models
```

## A modellek

A modellek listáját az AI-router `models.yaml`-ja tartalmazza. Belőle a `dwyt-airouter generate` két fájlt készít ide: `models.ini` és `downloads.json`.

```bash
# a models.ini a modellmappába kerül (a llama-server ott keresi)
scp AI-ROUTER-IP:/etc/dwyt-airouter/generated/models.ini  $DWYT_MODELS_DIR/
scp AI-ROUTER-IP:/etc/dwyt-airouter/generated/downloads.json /tmp/

export HF_TOKEN=hf_...                     # a beszélőfelismerő modellhez
dwyt-gpu download /tmp/downloads.json --dir $DWYT_MODELS_DIR --dry-run   # mi hiányzik, mennyi hely kell
dwyt-gpu download /tmp/downloads.json --dir $DWYT_MODELS_DIR
```

A letöltés megszakítás után újraindítva folytatódik. A meglévő fájlokat kihagyja.

## Ellenőrzés

```bash
dwyt-gpu check --dir $DWYT_MODELS_DIR --list /tmp/downloads.json
```

Minden sor „rendben” legyen: GPU, Docker GPU-hozzáféréssel, a két konténer (leállítva), a `models.ini` és a GGUF-fájlok. A teljes út (az AI-router indítja a konténert, a modell betölt) az AI-router leírásának végén ellenőrizhető.

## A Whisper-worker beállításai

A `whisper.env`-ben (Unraidon változóként):

| Változó | Alapérték | Mire való |
| --- | --- | --- |
| `HF_TOKEN` | – | Hugging Face-token a beszélőfelismeréshez |
| `WHISPER_MODELS` | `large-v3,large-v3-turbo` | A worker által elfogadott modellek; egyezzen a `models.yaml`-lal |
| `DEVICE` | `auto` | `auto`: GPU, ha látható, különben CPU; `cpu`: mindig CPU |
| `COMPUTE_TYPE` | automatikus | GPU-n `float16`, CPU-n `int8` |
| `BATCH_SIZE` | automatikus | GPU-n 8, CPU-n 4 |
| `CPU_THREADS` | fizikai magok | CPU-s futásnál |
| `DIARIZE_MODEL` | WhisperX alapérték | másik pyannote-modell |
| `ASR_CACHE_DAYS` | `14` | ennyi napig őrzi a beszélőfelismerés előtti átiratot (újrapróbáláskor onnan folytatja) |
| `MAX_AUDIO_MB` | `2048` | a fogadott hang legnagyobb mérete |

## Hibakeresés

| Jel | Ok és teendő |
| --- | --- |
| `dwyt-gpu check`: „Docker GPU-hozzáférés: nincs” | Az NVIDIA Container Toolkit hiányzik vagy nincs beállítva (`nvidia-ctk runtime configure --runtime=docker`, majd a Docker újraindítása). |
| A Whisper-munka „Hugging Face-token kell” hibával áll le | `HF_TOKEN` a `whisper.env`-ben, és a pyannote-modell feltételei elfogadva; utána `docker compose create --force-recreate whisper`. |
| A worker „CPU-n futott” figyelmeztetést ad | A konténer nem látja a GPU-t: Unraidon `--runtime=nvidia` és `NVIDIA_VISIBLE_DEVICES=all`, Linuxon a compose GPU-foglalása. |
| A llama-server nem indul: „models.ini” | A `models.ini` nincs a modellmappában, vagy régi; másold át újra a `generate` után. |
| Elfogyott a VRAM | Egyszerre csak egy konténer fusson. Ha más GPU-s konténer is van a gépen, vedd fel az AI-router `other_gpu_containers` listájába. |
