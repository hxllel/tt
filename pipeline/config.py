"""Carga de config.yaml con overrides y resolucion de rutas relativas a la raiz."""

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"

# Pesos de HuggingFace (CLIP, e5) dentro del proyecto, no en el perfil del usuario.
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides: dict = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _deep_merge(cfg, overrides)


def ruta(p) -> Path:
    """Convierte una ruta del config (relativa a la raiz) en absoluta."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def dir_video(cfg: dict, video_name: str) -> Path:
    return ruta(cfg["salidas_dir"]) / video_name


def videos_procesados(cfg: dict) -> list:
    base = ruta(cfg["salidas_dir"])
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if (d / "metadata.json").exists())
