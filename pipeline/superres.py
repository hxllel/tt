"""Superresolucion de recortes con spandrel (Real-ESRGAN, HAT, SwinIR...).

Solo se aplica al mejor recorte (rank 0) de cada persona y solo si es pequeno,
porque FastVLM/CLIP reescalan de todas formas y la SR es costosa. spandrel
reemplaza a basicsr, que ya no importa con torchvision >= 0.17."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.detect_track import _actualizar_stats, cargar_metadata


class SuperRes:
    def __init__(self, model_path, device: str, half: bool):
        from spandrel import ModelLoader
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No existe el modelo de superresolucion {model_path}. "
                                    "Corre: python download_models.py")
        self.model = ModelLoader().load_from_file(str(model_path))
        self.device = torch.device("cuda" if device == "cuda" else "cpu")
        self.model.to(self.device).eval()
        self.half = half and self.device.type == "cuda" and self.model.supports_half
        if self.half:
            self.model.half()
        self.scale = self.model.scale
        req = self.model.size_requirements
        self.multiple = max(1, getattr(req, "multiple_of", 1))
        self.minimum = max(1, getattr(req, "minimum", 1))

    @torch.inference_mode()
    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0)
        x = x.half() if self.half else x.float()
        x = x / 255.0
        h, w = x.shape[2:]
        th = max(self.minimum, -(-h // self.multiple) * self.multiple)
        tw = max(self.minimum, -(-w // self.multiple) * self.multiple)
        if th != h or tw != w:
            x = F.pad(x, (0, tw - w, 0, th - h), mode="replicate")
        y = self.model(x)[:, :, : h * self.scale, : w * self.scale]
        y = (y.float().clamp(0, 1) * 255.0).round().byte()
        return cv2.cvtColor(y[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)


def apply_to_video(out_dir, cfg: dict, sr: SuperRes, progress=None) -> dict:
    out_dir = Path(out_dir)
    meta = cargar_metadata(out_dir)
    max_h = cfg["superres"]["max_alto_px"]
    sr_dir = out_dir / "crops_sr"
    sr_dir.mkdir(exist_ok=True)
    objetivo = [e for e in meta["crops"]
                if e["rank"] == 0 and (e["bbox_yolo"][3] - e["bbox_yolo"][1]) < max_h]
    t0 = time.perf_counter()
    hechos, errores = 0, 0
    for i, e in enumerate(objetivo):
        img = cv2.imread(str(out_dir / e["crop_path"]))
        if img is None:
            errores += 1
            continue
        try:
            up = sr.upscale(img)
        except Exception as ex:  # un recorte fallido no detiene el lote
            print(f"  [SR] error en {e['crop_path']}: {ex}")
            errores += 1
            continue
        nombre = Path(e["crop_path"]).name
        cv2.imwrite(str(sr_dir / nombre), up)
        e["crop_sr_path"] = f"crops_sr/{nombre}"
        hechos += 1
        if progress and i % 10 == 0:
            progress(i / max(1, len(objetivo)), f"Superresolucion {i}/{len(objetivo)}")
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    stats = {"recortes_mejorados": hechos, "errores": errores,
             "tiempo_s": round(time.perf_counter() - t0, 2), "escala": sr.scale}
    _actualizar_stats(out_dir, {"superres": stats})
    return stats
