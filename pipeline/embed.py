"""Embeddings de texto (multilingual-e5) sobre las descripciones que genera FastVLM.

El sistema no usa CLIP: toda la interpretacion visual la hace FastVLM. Aqui solo
se vectoriza TEXTO —la descripcion generada y la consulta de la alerta— para
compararlas por similitud coseno (O.E.5 / RF-12). Esto cumple RNF-13: los
embeddings indexados son de texto, nunca de imagen ni biometricos."""

import time
from pathlib import Path

import numpy as np
import torch

from pipeline.detect_track import _actualizar_stats

CACHE = "emb_texto.npz"


class TextEmbedder:
    """intfloat/multilingual-e5-small con mean pooling (prefijos query:/passage:).

    Es multilingue, asi que empareja la alerta en espanol con la descripcion que
    FastVLM genera en ingles."""

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


def _cargar_cache(out_dir: Path) -> dict:
    p = out_dir / CACHE
    if not p.exists():
        return {}
    z = np.load(p)
    return {int(i): z["vecs"][k] for k, i in enumerate(z["ids"])}


def embeddings_descripciones(out_dir, embedder: TextEmbedder, descs: dict, crop_ids: list,
                             progress=None) -> dict:
    """{crop_id: vector} de las descripciones pedidas, cacheado en emb_texto.npz.
    Solo vectoriza las que faltan, asi que repetir una busqueda es instantaneo."""
    out_dir = Path(out_dir)
    por_id = _cargar_cache(out_dir)
    faltan = [c for c in crop_ids
              if c not in por_id and str(c) in descs and not descs[str(c)].get("error")]
    if faltan:
        if progress is not None:
            progress(0.5, f"Vectorizando {len(faltan)} descripciones")
        t0 = time.perf_counter()
        nuevos = embedder.encode([descs[str(c)]["texto_busqueda"] for c in faltan], "passage")
        por_id.update(zip(faltan, nuevos))
        np.savez(out_dir / CACHE, ids=np.array(list(por_id), dtype=np.int64),
                 vecs=np.stack(list(por_id.values())))
        _actualizar_stats(out_dir, {"embeddings_texto": {
            "modelo": embedder.nombre, "descripciones": len(por_id), "nuevas": len(faltan),
            "tiempo_s": round(time.perf_counter() - t0, 2)}})
    return por_id
