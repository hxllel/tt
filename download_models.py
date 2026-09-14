"""Descarga los pesos necesarios a models/ y ml-fastvlm/checkpoints/ (idempotente).

    python download_models.py                 # YOLO26 n/s, SR compacto, FastVLM 0.5B stage3, CLIP y e5
    python download_models.py --vlm 1.5b 7b   # checkpoints adicionales de FastVLM
    python download_models.py --hat           # (opcional) indica como obtener HAT
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from pipeline.config import ROOT, load_config, ruta

APPLE_CDN = "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_{}_stage{}.zip"
SR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"


def descargar(url: str, destino: Path):
    if destino.exists():
        print(f"  ya existe {destino.name}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    print(f"  descargando {url}")

    def avance(bloques, tam, total):
        if total > 0 and bloques % 200 == 0:
            print(f"    {min(100, bloques * tam * 100 // total)}%", end="\r")

    urllib.request.urlretrieve(url, tmp, avance)
    tmp.rename(destino)


def descargar_fastvlm(tam: str, stage: int):
    ckpts = ROOT / "ml-fastvlm" / "checkpoints"
    nombre = f"llava-fastvithd_{tam}_stage{stage}"
    if any((d / "config.json").exists() for d in [ckpts / nombre, *(ckpts / nombre).glob("*")]):
        print(f"  ya existe {nombre}")
        return
    zp = ckpts / f"{nombre}.zip"
    descargar(APPLE_CDN.format(tam, stage), zp)
    print(f"  descomprimiendo {zp.name}")
    with zipfile.ZipFile(zp) as z:
        z.extractall(ckpts)
    shutil.rmtree(ckpts / "__MACOSX", ignore_errors=True)
    zp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", nargs="*", default=["0.5b"], help="tamanos FastVLM stage3 a bajar")
    ap.add_argument("--yolo", nargs="*", default=["yolo26n", "yolo26s"])
    ap.add_argument("--sin-clip", action="store_true")
    ap.add_argument("--hat", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    modelos = ruta(cfg["modelos_dir"])
    modelos.mkdir(exist_ok=True)

    print("YOLO26:")
    from ultralytics.utils.downloads import attempt_download_asset
    for y in args.yolo:
        destino = modelos / f"{y}.pt"
        if not destino.exists():
            antiguo = ROOT / "ml-fastvlm" / "model_export" / f"{y}.pt"
            if antiguo.exists():
                shutil.copy(antiguo, destino)
            else:
                attempt_download_asset(str(destino))
        print(f"  {destino.name} {'ok' if destino.exists() else 'FALLO'}")

    print("Superresolucion:")
    descargar(SR_URL, modelos / "realesr-general-x4v3.pth")
    if args.hat:
        print("  HAT: descarga manual de HAT_SRx4_ImageNet-pretrain.pth desde https://github.com/XPixelGroup/HAT "
              "(Google Drive), colocalo en models/ y pon superres.modelo en config.yaml")

    print("FastVLM:")
    for tam in args.vlm:
        descargar_fastvlm(tam, 3)

    if not args.sin_clip:
        print("CLIP y embeddings de texto (HuggingFace):")
        from pipeline.embed import ClipEmbedder, TextEmbedder
        c = ClipEmbedder(cfg, "cpu", False)
        print(f"  {c.nombre} ok")
        t = TextEmbedder(cfg, "cpu", False)
        print(f"  {t.nombre} ok")
    print("Listo.")


if __name__ == "__main__":
    main()
