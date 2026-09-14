"""Deteccion de GPU/VRAM y seleccion del perfil de ejecucion.

    python -m pipeline.hardware
"""

import platform
from dataclasses import dataclass, asdict

import psutil

# Memoria aproximada (GB) de cada componente en fp16; se usa para estimar la carga.
VRAM_VLM_GB = {"0.5b": 1.4, "1.5b": 3.6, "7b": 15.5}
VRAM_YOLO_GB = {"yolo26n": 0.2, "yolo26s": 0.3, "yolo26m": 0.5, "yolo26l": 0.7}
VRAM_EXTRA_GB = 0.4 + 0.25 + 0.1   # CLIP B/16 + e5-small + SR compacto
VRAM_RESERVA_GB = 1.5              # activaciones del codificador a 1024 + KV cache


@dataclass
class HardwareInfo:
    device: str                 # "cuda" | "cpu"
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    compute_capability: str = ""
    cuda_version: str = ""
    torch_version: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0


@dataclass
class Profile:
    vlm: str
    yolo: str
    superres: bool
    vlm_batch: int
    clip_batch: int
    half: bool
    vram_estimada_gb: float = 0.0
    nota: str = ""


def detect(forzar: str = "auto") -> HardwareInfo:
    import torch

    info = HardwareInfo(
        device="cpu",
        torch_version=torch.__version__,
        cpu_name=platform.processor() or platform.machine(),
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
    )
    if forzar != "cpu" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        major, minor = torch.cuda.get_device_capability(0)
        info.device = "cuda"
        info.gpu_name = torch.cuda.get_device_name(0)
        info.vram_total_gb = round(total / 1e9, 2)
        info.vram_free_gb = round(free / 1e9, 2)
        info.compute_capability = f"{major}.{minor}"
        info.cuda_version = torch.version.cuda or ""
    elif forzar == "cuda":
        raise RuntimeError("Se forzo dispositivo=cuda pero torch no detecta una GPU CUDA.")
    return info


def choose_profile(info: HardwareInfo, vlm_disponibles=("0.5b", "1.5b", "7b")) -> Profile:
    """Elige modelos y tamanos de lote segun la VRAM libre (o CPU)."""
    if info.device == "cpu":
        # SR compacta cuesta ~0.1 s por recorte en CPU y mejora mucho los recortes chicos
        p = Profile(vlm="0.5b", yolo="yolo26n", superres=True, vlm_batch=1, clip_batch=32,
                    half=False, nota="CPU: FastVLM en fp32; la primera descripcion tarda ~2 min en calentar")
    else:
        free = info.vram_free_gb
        if free < 4:
            p = Profile("0.5b", "yolo26n", True, 4, 128, True)
        elif free < 10:
            p = Profile("1.5b", "yolo26s", True, 8, 256, True)
        elif free < 20:
            p = Profile("1.5b", "yolo26m", True, 16, 256, True)
        else:
            p = Profile("7b", "yolo26m", True, 16, 512, True)

    # Si el checkpoint recomendado no existe, baja al mas grande disponible que quepa.
    orden = ["7b", "1.5b", "0.5b"]
    if p.vlm not in vlm_disponibles:
        candidatos = [m for m in orden[orden.index(p.vlm):] if m in vlm_disponibles]
        if not candidatos:  # solo hay modelos mas grandes que el recomendado
            candidatos = [m for m in reversed(orden) if m in vlm_disponibles]
        if candidatos:
            p.nota = (p.nota + f" | checkpoint {p.vlm} no disponible, se usa {candidatos[0]}").strip(" |")
            p.vlm = candidatos[0]

    p.vram_estimada_gb = round(estimate_vram_gb(p), 2)
    if info.device == "cuda" and p.vram_estimada_gb > info.vram_free_gb:
        p.nota = (p.nota + " | la carga estimada supera la VRAM libre; se reducira el lote si hay OOM").strip(" |")
    return p


def estimate_vram_gb(p: Profile) -> float:
    return VRAM_VLM_GB.get(p.vlm, 3.6) + VRAM_YOLO_GB.get(p.yolo, 0.3) + VRAM_EXTRA_GB + VRAM_RESERVA_GB


def is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def run_batched(fn, items: list, batch: int, on_batch_done=None):
    """Aplica fn a items en lotes; si hay OOM divide el lote a la mitad y reintenta.

    Devuelve (resultados, lote_final)."""
    resultados = []
    i = 0
    while i < len(items):
        chunk = items[i:i + batch]
        try:
            resultados.extend(fn(chunk))
        except (RuntimeError, MemoryError) as e:
            if not is_oom(e) or batch == 1:
                raise
            batch = max(1, batch // 2)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            continue
        i += len(chunk)
        if on_batch_done:
            on_batch_done(i, len(items))
    return resultados, batch


def resumen(info: HardwareInfo, p: Profile) -> str:
    if info.device == "cuda":
        hw = (f"**GPU:** {info.gpu_name} (compute {info.compute_capability}, CUDA {info.cuda_version})  \n"
              f"**VRAM:** {info.vram_free_gb:.2f} GB libres de {info.vram_total_gb:.2f} GB")
    else:
        hw = "**GPU CUDA:** no detectada, se ejecuta en CPU"
    return (f"{hw}  \n**CPU:** {info.cpu_name} ({info.cpu_cores} nucleos fisicos), RAM {info.ram_gb} GB  \n"
            f"**torch:** {info.torch_version}  \n\n"
            f"**Perfil:** FastVLM {p.vlm} | {p.yolo} | superresolucion {'si' if p.superres else 'no'} | "
            f"lote VLM {p.vlm_batch} | lote CLIP {p.clip_batch} | {'fp16' if p.half else 'fp32'}  \n"
            f"**Carga estimada:** {p.vram_estimada_gb} GB" + (f"  \n_{p.nota}_" if p.nota else ""))


if __name__ == "__main__":
    info = detect()
    perfil = choose_profile(info)
    print(asdict(info))
    print(asdict(perfil))
