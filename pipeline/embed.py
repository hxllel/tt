"""Embeddings: CLIP (imagen <-> texto alineados) y texto multilingue (e5).

OJO: el vector interno de FastVLM no esta alineado con texto; para comparar un
recorte contra la descripcion de una alerta por coseno se necesita un modelo
entrenado contrastivamente como CLIP/SigLIP."""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline.detect_track import _actualizar_stats, cargar_metadata
from pipeline.hardware import run_batched


class ClipEmbedder:
    def __init__(self, cfg: dict, device: str, half: bool):
        import open_clip
        b = cfg["busqueda"]
        self.device = torch.device("cuda" if device == "cuda" else "cpu")
        precision = "fp16" if (half and self.device.type == "cuda") else "fp32"
        intentos = [(b["clip_modelo"], b["clip_pretrained"]), tuple(b["clip_respaldo"])]
        ultimo_error = None
        for nombre, pre in intentos:
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    nombre, pretrained=pre, device=self.device, precision=precision)
                self.tokenizer = open_clip.get_tokenizer(nombre)
                self.nombre = f"{nombre}/{pre}"
                break
            except Exception as e:
                print(f"  [CLIP] no se pudo cargar {nombre}/{pre}: {e}")
                ultimo_error = e
        else:
            raise RuntimeError(f"No se pudo cargar ningun modelo CLIP: {ultimo_error}")
        self.model.eval()
        self.dtype = torch.float16 if precision == "fp16" else torch.float32

    @torch.inference_mode()
    def encode_images(self, paths: list, batch: int, progress=None):
        def lote(chunk):
            x = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in chunk])
            f = self.model.encode_image(x.to(self.device, dtype=self.dtype))
            return list(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())

        cb = (lambda i, n: progress(i / n, f"CLIP: {i}/{n} recortes")) if progress else None
        vecs, batch = run_batched(lote, list(paths), batch, cb)
        return np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 1), np.float32), batch

    @torch.inference_mode()
    def encode_texts(self, textos: list) -> np.ndarray:
        tok = self.tokenizer(textos).to(self.device)
        f = self.model.encode_text(tok)
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float32)


class TextEmbedder:
    """intfloat/multilingual-e5-small con mean pooling (prefijos query:/passage:)."""

    def __init__(self, cfg: dict, device: str, half: bool):
        from transformers import AutoModel, AutoTokenizer
        name = cfg["busqueda"]["texto_modelo"]
        self.device = torch.device("cuda" if device == "cuda" else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        if half and self.device.type == "cuda":
            self.model.half()
        self.nombre = name

    @torch.inference_mode()
    def encode(self, textos: list, prefijo: str, batch: int = 64) -> np.ndarray:
        salida = []
        for i in range(0, len(textos), batch):
            enc = self.tokenizer([f"{prefijo}: {t}" for t in textos[i:i + batch]], padding=True,
                                 truncation=True, max_length=256, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            salida.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(salida).astype(np.float32) if salida else np.zeros((0, 1), np.float32)


def embed_video_clip(out_dir, clip: ClipEmbedder, batch: int, progress=None) -> np.ndarray:
    """Embeddings CLIP de todos los recortes -> emb_clip.npy (fila i = crop_id i)."""
    out_dir = Path(out_dir)
    meta = cargar_metadata(out_dir)
    paths = [out_dir / (e.get("crop_sr_path") or e["crop_path"]) for e in meta["crops"]]
    t0 = time.perf_counter()
    emb, batch = clip.encode_images(paths, batch, progress)
    np.save(out_dir / "emb_clip.npy", emb)
    dur = time.perf_counter() - t0
    _actualizar_stats(out_dir, {"clip": {"modelo": clip.nombre, "recortes": len(paths), "lote_final": batch,
                                         "tiempo_s": round(dur, 2),
                                         "recortes_por_s": round(len(paths) / dur, 1) if dur else 0}})
    return emb
