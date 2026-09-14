"""
Etapa 1: Extraccion de personas con YOLO26 + Tracking (CLI)
-----------------------------------------------------------
Detecta y trackea personas y guarda, por cada persona unica (track_id), sus
mejores recortes en salidas/<video>/crops/ junto con metadata.json y stats.json.
La logica vive en pipeline/detect_track.py (tambien la usa app.py).

Uso:
    python extract_bounding_boxes.py videos/Video_51.mov [otro.mp4 ...]
    python extract_bounding_boxes.py video.mp4 --dispositivo cuda --intervalo 5 --modelo yolo26s
"""

import argparse
import json

from pipeline.config import load_config
from pipeline.detect_track import process_video
from pipeline.hardware import choose_profile, detect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config")
    ap.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--intervalo", type=int)
    ap.add_argument("--modelo", help="yolo26n | yolo26s | yolo26m | yolo26l")
    ap.add_argument("--imgsz", type=int)
    args = ap.parse_args()

    det = {k: v for k, v in {"intervalo_frames": args.intervalo, "modelo": args.modelo,
                             "imgsz": args.imgsz}.items() if v is not None}
    over = {"deteccion": det}
    if args.dispositivo:
        over["dispositivo"] = args.dispositivo
    cfg = load_config(args.config, over)
    info = detect(cfg["dispositivo"])
    perfil = choose_profile(info)

    for v in args.videos:
        meta = process_video(v, cfg, info.device, perfil.yolo,
                             progress=lambda f, msg: print(f"  {msg}", end="\r"))
        print(f"\n'{meta['video']}': {len(meta['tracks'])} personas, {len(meta['crops'])} recortes")
    print(json.dumps({"dispositivo": info.device, "detector": meta["detector"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
