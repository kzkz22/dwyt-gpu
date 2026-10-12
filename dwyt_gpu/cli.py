"""dwyt-gpu: tools for the GPU host.

  dwyt-gpu download LIST --dir DIR [--dry-run]   download the models listed in downloads.json
  dwyt-gpu check [--dir DIR] [--list LIST]        driver, Docker GPU access, containers, model folder
  dwyt-gpu token --dir DIR [--remove]            store the Hugging Face token in the model folder
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from . import __version__, check, download


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="dwyt-gpu", description="A dwyt GPU-gépének eszközei.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="modellek letöltése a downloads.json szerint")
    d.add_argument("list", help="a dwyt-airouter generate által készített downloads.json")
    d.add_argument("--dir", default=os.environ.get("DWYT_MODELS_DIR"), help="modellmappa (DWYT_MODELS_DIR)")
    d.add_argument("--dry-run", action="store_true", help="csak megmutatja, mi hiányzik")
    c = sub.add_parser("check", help="a GPU-gép ellenőrzése")
    c.add_argument("--dir", default=os.environ.get("DWYT_MODELS_DIR"))
    c.add_argument("--list", help="downloads.json: a GGUF-fájlok meglétét is nézi")
    t = sub.add_parser("token", help="a Hugging Face-token tárolása a modellmappában (hf/token)")
    t.add_argument("--dir", default=os.environ.get("DWYT_MODELS_DIR"), help="modellmappa (DWYT_MODELS_DIR)")
    t.add_argument("--remove", action="store_true", help="a tárolt token törlése")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "download":
            if not args.dir:
                parser.error("add meg a modellmappát: --dir vagy DWYT_MODELS_DIR")
            return download.run(args.list, args.dir, args.dry_run)
        if args.cmd == "token":
            if not args.dir:
                parser.error("add meg a modellmappát: --dir vagy DWYT_MODELS_DIR")
            return _token(Path(args.dir), args.remove)
        return check.run(args.dir, args.list)
    except download.DownloadError as exc:
        print(f"Hiba: {exc}", file=sys.stderr)
        return 1


def _token(models_dir: Path, remove: bool) -> int:
    path = models_dir / download.TOKEN_FILE
    if remove:
        path.unlink(missing_ok=True)
        print(f"Törölve: {path}")
        return 0
    token = getpass.getpass("Hugging Face-token (beillesztés után Enter; nem látszik): ").strip()
    if not token:
        print("Nem adtál meg tokent; semmi nem változott.", file=sys.stderr)
        return 1
    name = download.verify_token(token)
    print(f"Elmentve ide: {download.save_token(models_dir, token)} (csak a root olvashatja; fiók: {name})")
    print("A dwyt-gpu download mostantól ezt használja, ha a HF_TOKEN nincs beállítva.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
