"""
Superresolucion de recortes (CLI)
---------------------------------
Aplica superresolucion (Real-ESRGAN compacto por defecto, o un checkpoint HAT)
solo al mejor recorte de cada persona cuando es pequeno. Usa spandrel en lugar
de basicsr (basicsr 1.4.2 falla con torchvision >= 0.17). Logica en
pipeline/superres.py.

Uso:
    python improve_quality.py Video_51
    python improve_quality.py Video_51 --modelo HAT_SRx4_ImageNet-pretrain.pth --max-alto 200
"""

import argparse

from pipeline.config import dir_video, load_config, ruta
from pipeline.hardware import choose_profile, detect
from pipeline.superres import SuperRes, apply_to_video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="nombre del video en salidas/")
    ap.add_argument("--modelo", help="archivo dentro de models/")
    ap.add_argument("--max-alto", type=int)
    ap.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    sr_cfg = {k: v for k, v in {"modelo": args.modelo, "max_alto_px": args.max_alto}.items() if v}
    over = {"superres": sr_cfg}
    if args.dispositivo:
        over["dispositivo"] = args.dispositivo
    cfg = load_config(overrides=over)
    info = detect(cfg["dispositivo"])
    sr = SuperRes(ruta(cfg["modelos_dir"]) / cfg["superres"]["modelo"], info.device, choose_profile(info).half)
    print(apply_to_video(dir_video(cfg, args.video), cfg, sr))


if __name__ == "__main__":
    main()
