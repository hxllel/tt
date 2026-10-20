"""Carga de config.yaml con overrides y resolucion de rutas relativas a la raiz."""

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
# Ajustes de ESTA maquina, generados por init_entorno.py. No se versiona.
LOCAL_CONFIG = ROOT / "config.local.yaml"

# Pesos de HuggingFace (modelo de embeddings de texto) dentro del proyecto.
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _leer(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path=None, overrides: dict = None) -> dict:
    """config.yaml -> config.local.yaml (si existe) -> overrides del llamador."""
    cfg = _leer(path or DEFAULT_CONFIG)
    if path is None and LOCAL_CONFIG.exists():
        cfg = _deep_merge(cfg, _leer(LOCAL_CONFIG))
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
