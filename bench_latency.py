"""Benchmark de latencia (Capa 4 del Plan de Validacion).

Mide sobre un recorte fijo, con el hardware documentado en el CSV:
  - carga del modelo (= costo por recorte del esquema anterior con subprocess)
  - TTFT, tiempo de generacion, tokens/s y recortes/s
  - barrido de resolucion del codificador, tamano de lote y max_new_tokens
  - FPS de YOLO26 sobre el video (opcional)

    python bench_latency.py --crop salidas/Video_51/crops/track_00001_r0_f120.jpg
    python bench_latency.py --device cuda --batches 1 8 16 --resoluciones 1024 768 512 --video videos/Video_51.mov
"""

import argparse
import csv
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline.config import ROOT, load_config
from pipeline.hardware import choose_profile, detect


def medir_vlm(cfg, info, vlm, crop, resoluciones, batches, tokens_list, reps, warmup):
    from pipeline.describe import Describer
    filas = []
    img = Image.open(crop).convert("RGB")
    desc = None
    for res in resoluciones:
        cfg["vlm"]["resolucion"] = res
        if desc is None:
            desc = Describer(cfg, info.device, vlm, 1)
            carga = desc.load_s
            print(f"Carga de FastVLM {vlm}: {carga:.2f} s (antes se pagaba por CADA recorte)")
        desc.image_processor.crop_size = {"height": res, "width": res}
        desc.image_processor.size = {"shortest_edge": res}
        for b in batches:
            for mt in tokens_list:
                imgs = [img] * b
                try:
                    for _ in range(warmup):
                        desc.generate(imgs, mt)
                    recs = [desc.generate(imgs, mt)[0] for _ in range(reps)]
                except RuntimeError as e:
                    print(f"  res={res} lote={b}: {e}")
                    break
                ttft = np.array([r["ttft_s"] for r in recs])
                gen = np.array([r["gen_s"] for r in recs])
                toks = np.array([r["tokens"] for r in recs])
                fila = {"componente": f"fastvlm_{vlm}", "resolucion": res, "lote": b, "max_new_tokens": mt,
                        "ttft_media_s": ttft.mean(), "ttft_std_s": ttft.std(), "gen_media_s": gen.mean(),
                        "gen_std_s": gen.std(), "tokens_media": toks.mean(),
                        "tokens_por_s": (toks.mean() * b) / gen.mean(), "recortes_por_s": b / gen.mean(),
                        "carga_modelo_s": carga, "repeticiones": reps}
                filas.append(fila)
                print(f"  res={res:4d} lote={b:2d} max_tok={mt:3d} | TTFT {ttft.mean():.3f}±{ttft.std():.3f} s | "
                      f"gen {gen.mean():.2f} s | {fila['recortes_por_s']:.2f} recortes/s | salida {toks.mean():.0f} tok")
        print(f"  ejemplo de salida (res={res}): {desc.generate([img])[0]['raw'][:200]}")
    return filas


def medir_yolo(cfg, info, perfil, video, n_frames):
    from pipeline.detect_track import cargar_detector, resolver_detector, yolo_kwargs
    from pipeline.video_io import iter_frames
    path = resolver_detector(cfg, info.device, perfil.yolo)
    model = cargar_detector(path)
    kw = dict(classes=[0], conf=cfg["deteccion"]["conf_tracker"], verbose=False, **yolo_kwargs(cfg, info.device, path))
    frames = [f for _, _, f in itertools.islice(iter_frames(video, 1), n_frames)]
    for f in frames[:3]:
        model.predict(f, **kw)
    t0 = time.perf_counter()
    for f in frames:
        model.predict(f, **kw)
    if info.device == "cuda":
        torch.cuda.synchronize()
    dur = time.perf_counter() - t0
    print(f"YOLO {path.name} imgsz={kw['imgsz']}: {len(frames) / dur:.1f} FPS")
    return [{"componente": f"yolo_{path.name}", "resolucion": kw["imgsz"], "lote": 1,
             "fps": len(frames) / dur, "repeticiones": len(frames)}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--vlm", help="0.5b | 1.5b | 7b (por defecto el del perfil)")
    ap.add_argument("--crop", help="recorte fijo; por defecto el primero de salidas/")
    ap.add_argument("--resoluciones", nargs="+", type=int, default=[1024, 768, 512])
    ap.add_argument("--batches", nargs="+", type=int, default=[1])
    ap.add_argument("--max-new-tokens", nargs="+", type=int, default=[96, 256])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--video", help="si se indica, mide tambien FPS de YOLO")
    ap.add_argument("--frames-yolo", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(overrides={"dispositivo": args.device, "vlm": {}})
    info = detect(cfg["dispositivo"])
    from pipeline.describe import checkpoints_disponibles
    perfil = choose_profile(info, checkpoints_disponibles(cfg))
    if info.device == "cpu":
        torch.set_num_threads(info.cpu_cores)
    crop = args.crop or next(iter(sorted((ROOT / "salidas").glob("*/crops/*.jpg"))), None)
    if crop is None:
        raise SystemExit("No hay recortes: procesa un video primero o pasa --crop")

    filas = []
    if args.video:
        filas += medir_yolo(cfg, info, perfil, args.video, args.frames_yolo)
    filas += medir_vlm(cfg, info, args.vlm or perfil.vlm, crop, args.resoluciones, args.batches,
                       args.max_new_tokens, args.reps, args.warmup)

    hw = {"dispositivo": info.device, "gpu": info.gpu_name, "vram_total_gb": info.vram_total_gb,
          "cuda": info.cuda_version, "cpu": info.cpu_name, "nucleos": info.cpu_cores, "torch": info.torch_version}
    out = ROOT / "salidas" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"latencia_{info.device}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    campos = list(dict.fromkeys(k for f in filas for k in f)) + list(hw)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        for fila in filas:
            wr.writerow({**fila, **hw})
    print(f"CSV: {path}")


if __name__ == "__main__":
    main()
