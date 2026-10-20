"""
Diagnostico e inicializacion del entorno (TT 2027-A048)
=======================================================
Revisa que la maquina tenga todo lo necesario, crea las carpetas de trabajo,
detecta la GPU y deja escrito un `config.local.yaml` con los valores concretos
que le convienen a ESTE equipo (dispositivo, detector, tamano de FastVLM, lote,
superresolucion). `config.yaml` no se toca: es el valor por defecto del proyecto
y `config.local.yaml` solo lo sobreescribe en esta maquina.

Este archivo solo importa la biblioteca estandar al cargarse, asi que se puede
ejecutar ANTES de instalar las dependencias para saber que falta.

Uso:
    python init_entorno.py                 # solo diagnostico
    python init_entorno.py --inicializar   # + crea carpetas y escribe config.local.yaml
    python init_entorno.py --descargar     # + descarga los modelos que falten
    python init_entorno.py --todo          # las tres cosas
    python init_entorno.py --json          # salida para scripts (no imprime el reporte)

Codigo de salida: 0 si el entorno esta listo, 1 si falta algo critico.
"""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY_MINIMO = (3, 10)
DISCO_MINIMO_GB = 8

# (modulo que se importa, paquete de pip, para que sirve, es critico)
PAQUETES = [
    ("torch", "torch==2.6.0", "inferencia", True),
    ("torchvision", "torchvision==0.21.0", "inferencia", True),
    ("ultralytics", "ultralytics==8.4.129", "deteccion YOLO26", True),
    ("cv2", "opencv-python==4.11.0.86", "lectura de video", True),
    ("lap", "lap==0.5.13", "asociacion de ByteTrack", True),
    ("transformers", "transformers==4.48.3", "FastVLM y embeddings", True),
    ("llava", "-e ./ml-fastvlm (con --no-deps)", "FastVLM", True),
    ("yaml", "pyyaml", "configuracion", True),
    ("psutil", "psutil", "deteccion de hardware", True),
    ("numpy", "numpy==1.26.4", "calculo", True),
    ("gradio", "gradio>=5.35,<6", "interfaz grafica", True),
    ("spandrel", "spandrel", "superresolucion", False),
    ("pandas", "pandas", "tablas de la interfaz", False),
    ("matplotlib", "matplotlib", "graficas de metricas", False),
    ("motmetrics", "motmetrics", "metricas de tracking (MOTA/IDF1)", False),
    ("pytest", "pytest", "pruebas", False),
]

# Las descripciones de FastVLM son lo unico que hace buscable un video, asi que
# el checkpoint del VLM es critico; el resto se puede bajar despues.
MODELOS = [
    ("models/yolo26n.pt", "detector ligero", True),
    ("models/yolo26s.pt", "detector recomendado en GPU", False),
    ("models/realesr-general-x4v3.pth", "superresolucion", False),
]
CHECKPOINTS_VLM = {
    "0.5b": "ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3",
    "1.5b": "ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage2",
    "7b": "ml-fastvlm/checkpoints/llava-fastvithd_7b_stage3",
}

OK, FALTA, AVISO = "ok", "falta", "aviso"
ICONO = {OK: "[ok]   ", FALTA: "[FALTA]", AVISO: "[aviso]"}


class Reporte:
    def __init__(self):
        self.filas = []      # (estado, titulo, detalle, arreglo)
        self.datos = {}

    def add(self, estado, titulo, detalle="", arreglo=""):
        self.filas.append((estado, titulo, detalle, arreglo))

    @property
    def faltantes(self):
        return [f for f in self.filas if f[0] == FALTA]

    def imprimir(self):
        ancho = max(len(f[1]) for f in self.filas) + 2
        seccion = None
        for estado, titulo, detalle, _ in self.filas:
            nueva = titulo.split(":")[0] if ":" in titulo else None
            if nueva and nueva != seccion:
                seccion = nueva
            print(f"{ICONO[estado]} {titulo.ljust(ancho)} {detalle}")
        arreglos = [f[3] for f in self.filas if f[0] != OK and f[3]]
        if arreglos:
            print("\nComo arreglarlo:")
            for a in dict.fromkeys(arreglos):   # sin repetir, en orden
                print(f"  {a}")


# --------------------------------------------------------------- deteccion
def hay(modulo: str) -> bool:
    try:
        return importlib.util.find_spec(modulo) is not None
    except (ImportError, ValueError):
        return False


def info_sistema() -> dict:
    libre = shutil.disk_usage(ROOT).free / 1e9
    d = {
        "so": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "python_ok": sys.version_info >= PY_MINIMO,
        "ejecutable": sys.executable,
        "entorno_conda": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "disco_libre_gb": round(libre, 1),
    }
    if hay("psutil"):
        import psutil
        d["cpu_nucleos"] = psutil.cpu_count(logical=False)
        d["cpu_hilos"] = psutil.cpu_count(logical=True)
        d["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    d["cpu"] = platform.processor() or platform.machine()
    return d


def info_gpu() -> dict:
    """GPU segun torch; si torch no esta, al menos lo que diga nvidia-smi."""
    g = {"disponible": False, "fuente": "ninguna"}
    if hay("torch"):
        import torch
        g["torch"] = torch.__version__
        g["torch_cuda_compilado"] = torch.version.cuda
        if torch.cuda.is_available():
            libre, total = torch.cuda.mem_get_info(0)
            mayor, menor = torch.cuda.get_device_capability(0)
            g.update(disponible=True, fuente="torch", nombre=torch.cuda.get_device_name(0),
                     vram_total_gb=round(total / 1e9, 2), vram_libre_gb=round(libre / 1e9, 2),
                     capacidad=f"{mayor}.{menor}")
            return g
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            salida = subprocess.run(
                [smi, "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
            if salida.returncode == 0 and salida.stdout.strip():
                nombre, total, libre, driver = [x.strip() for x in salida.stdout.splitlines()[0].split(",")]
                g.update(disponible=True, fuente="nvidia-smi", nombre=nombre, driver=driver,
                         vram_total_gb=round(float(total) / 1024, 2),
                         vram_libre_gb=round(float(libre) / 1024, 2))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return g


def perfil(gpu: dict) -> dict:
    """Valores concretos recomendados para esta maquina.

    Usa pipeline.hardware cuando se puede (misma logica que en ejecucion) y, si
    las dependencias aun no estan, replica la tabla por VRAM."""
    disponibles = [n for n, p in CHECKPOINTS_VLM.items() if checkpoint_ok(ROOT / p)]
    if hay("torch") and hay("psutil") and hay("yaml"):
        try:
            sys.path.insert(0, str(ROOT))
            from pipeline.hardware import choose_profile, detect
            p = choose_profile(detect(), disponibles or ("0.5b",))
            return {"dispositivo": "cuda" if gpu.get("fuente") == "torch" and gpu["disponible"] else "cpu",
                    "vlm": p.vlm, "yolo": p.yolo, "superres": p.superres, "vlm_batch": p.vlm_batch,
                    "half": p.half, "nota": p.nota, "fuente": "pipeline.hardware"}
        except Exception as e:                       # entorno a medio instalar
            print(f"  (no se pudo usar pipeline.hardware: {e})", file=sys.stderr)
    libre = gpu.get("vram_libre_gb", 0) if gpu.get("disponible") else 0
    if not gpu.get("disponible"):
        p = dict(dispositivo="cpu", vlm="0.5b", yolo="yolo26n", superres=True, vlm_batch=1, half=False)
    elif libre < 4:
        p = dict(dispositivo="cuda", vlm="0.5b", yolo="yolo26n", superres=True, vlm_batch=4, half=True)
    elif libre < 10:
        p = dict(dispositivo="cuda", vlm="1.5b", yolo="yolo26s", superres=True, vlm_batch=8, half=True)
    elif libre < 20:
        p = dict(dispositivo="cuda", vlm="1.5b", yolo="yolo26m", superres=True, vlm_batch=16, half=True)
    else:
        p = dict(dispositivo="cuda", vlm="7b", yolo="yolo26m", superres=True, vlm_batch=16, half=True)
    if disponibles and p["vlm"] not in disponibles:
        orden = ["7b", "1.5b", "0.5b"]
        candidatos = [m for m in orden[orden.index(p["vlm"]):] if m in disponibles] or disponibles
        p["nota"] = f"checkpoint {p['vlm']} no descargado, se usa {candidatos[0]}"
        p["vlm"] = candidatos[0]
    p["fuente"] = "tabla por VRAM"
    return p


def checkpoint_ok(ruta: Path) -> bool:
    """El zip de Apple a veces trae el checkpoint dentro de otra carpeta."""
    if (ruta / "config.json").exists():
        return True
    return ruta.exists() and any(
        (s / "config.json").exists() for s in ruta.iterdir() if s.is_dir() and "__MACOSX" not in s.name)


# --------------------------------------------------------------- revision
def revisar() -> Reporte:
    r = Reporte()
    sis = info_sistema()
    gpu = info_gpu()
    r.datos = {"sistema": sis, "gpu": gpu}

    r.add(OK if sis["python_ok"] else FALTA, "Python",
          f"{sis['python']} ({sis['ejecutable']})",
          f"Se necesita Python >= {'.'.join(map(str, PY_MINIMO))}: conda env create -f environment.yml")
    entorno = sis["entorno_conda"]
    r.add(OK if entorno == "tt-a048" else AVISO, "Entorno conda",
          entorno or "ninguno activo",
          "conda activate tt-a048" if entorno != "tt-a048" else "")
    r.add(OK, "Sistema", f"{sis['so']} | {sis['cpu']}")
    if "ram_gb" in sis:
        r.add(OK, "CPU/RAM", f"{sis['cpu_nucleos']} nucleos fisicos, {sis['ram_gb']} GB de RAM")
    r.add(OK if sis["disco_libre_gb"] >= DISCO_MINIMO_GB else AVISO, "Disco libre",
          f"{sis['disco_libre_gb']} GB",
          f"Deja al menos {DISCO_MINIMO_GB} GB: los modelos pesan varios GB")

    # --- GPU
    if gpu["disponible"] and gpu["fuente"] == "torch":
        r.add(OK, "GPU", f"{gpu['nombre']} | {gpu['vram_libre_gb']} GB libres de "
                         f"{gpu['vram_total_gb']} GB | CUDA {gpu.get('torch_cuda_compilado')}")
    elif gpu["disponible"]:
        r.add(FALTA, "GPU", f"{gpu['nombre']} detectada por nvidia-smi, pero torch no la usa",
              "pip install torch==2.6.0 torchvision==0.21.0 "
              "--index-url https://download.pytorch.org/whl/cu124")
    elif hay("torch"):
        r.add(AVISO, "GPU", "sin GPU NVIDIA: todo correra en CPU (describir sera lento)")
    else:
        r.add(AVISO, "GPU", "no se puede saber todavia (falta torch)")

    # --- paquetes
    faltan_criticos, faltan_opcionales = [], []
    for modulo, pip_name, para_que, critico in PAQUETES:
        if hay(modulo):
            continue
        (faltan_criticos if critico else faltan_opcionales).append((modulo, pip_name, para_que))
    if faltan_criticos:
        for modulo, pip_name, para_que in faltan_criticos:
            # llava no esta en requirements.txt: se instala desde el repo de Apple
            arreglo = ("pip install --no-deps -e ./ml-fastvlm" if modulo == "llava"
                       else "pip install -r requirements.txt")
            r.add(FALTA, f"Paquete {modulo}", para_que, arreglo)
    else:
        r.add(OK, "Paquetes criticos", f"{len(PAQUETES) - len(faltan_opcionales)} instalados")
    for modulo, _, para_que in faltan_opcionales:
        r.add(AVISO, f"Paquete {modulo}", f"opcional: {para_que}", "pip install -r requirements.txt")

    # --- codigo de Apple
    r.add(OK if (ROOT / "ml-fastvlm" / "predict.py").exists() else FALTA, "Codigo ml-fastvlm",
          "presente" if (ROOT / "ml-fastvlm" / "predict.py").exists() else "no esta",
          "git clone https://github.com/apple/ml-fastvlm.git ml-fastvlm")

    # --- modelos
    for rel, para_que, critico in MODELOS:
        existe = (ROOT / rel).exists()
        r.add(OK if existe else (FALTA if critico else AVISO), f"Modelo {Path(rel).name}",
              para_que if existe else f"no descargado ({para_que})",
              "python init_entorno.py --descargar")
    vlm_ok = [n for n, p in CHECKPOINTS_VLM.items() if checkpoint_ok(ROOT / p)]
    r.add(OK if vlm_ok else FALTA, "Checkpoints FastVLM",
          ", ".join(vlm_ok) if vlm_ok else "ninguno descargado",
          "python init_entorno.py --descargar")
    r.datos["checkpoints_vlm"] = vlm_ok

    # --- carpetas
    faltan_dirs = [d for d in ("models", "videos", "salidas") if not (ROOT / d).is_dir()]
    r.add(OK if not faltan_dirs else AVISO, "Carpetas de trabajo",
          "models/ videos/ salidas/" if not faltan_dirs else f"faltan: {', '.join(faltan_dirs)}",
          "python init_entorno.py --inicializar")

    # --- perfil resultante
    p = perfil(gpu)
    r.datos["perfil"] = p
    r.add(OK, "Perfil para este equipo",
          f"{p['dispositivo']} | FastVLM {p['vlm']} | {p['yolo']} | lote {p['vlm_batch']} | "
          f"{'fp16' if p['half'] else 'fp32'} | SR {'si' if p['superres'] else 'no'}"
          + (f" | {p['nota']}" if p.get("nota") else ""))
    return r


# --------------------------------------------------------------- acciones
def inicializar(r: Reporte) -> list:
    """Crea las carpetas y escribe config.local.yaml con los valores de esta maquina."""
    hechos = []
    for d in ("models", "videos", "salidas"):
        ruta = ROOT / d
        if not ruta.is_dir():
            ruta.mkdir(parents=True, exist_ok=True)
            hechos.append(f"creada carpeta {d}/")
    p = r.datos["perfil"]
    destino = ROOT / "config.local.yaml"
    contenido = f"""# Generado por init_entorno.py para este equipo. NO se sube a git.
# Sobreescribe lo que haga falta de config.yaml; borra este archivo para volver
# a los valores automaticos del proyecto.
# Equipo: {r.datos['sistema']['so']} | {r.datos['gpu'].get('nombre', 'sin GPU NVIDIA')}
dispositivo: {p['dispositivo']}

deteccion:
  modelo: {p['yolo']}
  imgsz: {960 if p['dispositivo'] == 'cuda' else 640}

superres:
  activar: {'true' if p['superres'] else 'false'}

vlm:
  modelo: {p['vlm']}
  batch: {p['vlm_batch']}
"""
    destino.write_text(contenido, encoding="utf-8")
    hechos.append(f"escrito {destino.name}")
    return hechos


def descargar(vlm: str) -> int:
    cmd = [sys.executable, str(ROOT / "download_models.py"), "--vlm", vlm]
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inicializar", action="store_true",
                    help="crea las carpetas y escribe config.local.yaml")
    ap.add_argument("--descargar", action="store_true", help="descarga los modelos que falten")
    ap.add_argument("--todo", action="store_true", help="--inicializar y --descargar")
    ap.add_argument("--json", action="store_true", help="salida JSON en vez del reporte")
    args = ap.parse_args()
    inicializar_f = args.inicializar or args.todo
    descargar_f = args.descargar or args.todo

    r = revisar()
    if not args.json:
        print(f"\nEntorno del proyecto TT 2027-A048  ({ROOT})\n")
        r.imprimir()

    if descargar_f:
        if not hay("ultralytics") or not hay("yaml"):
            print("\nNo se pueden descargar los modelos sin las dependencias. "
                  "Instala primero: pip install -r requirements.txt")
            return 1
        if descargar(r.datos["perfil"]["vlm"]) != 0:
            print("\nLa descarga fallo; revisa la conexion y reintenta.")
            return 1
        r = revisar()        # vuelve a medir con los modelos ya presentes

    if inicializar_f:
        for h in inicializar(r):
            print(f"  {h}")

    if args.json:
        print(json.dumps({**r.datos,
                          "faltantes": [f[1] for f in r.faltantes],
                          "listo": not r.faltantes}, indent=1, ensure_ascii=False))
        return 0 if not r.faltantes else 1

    if r.faltantes:
        print(f"\nFaltan {len(r.faltantes)} cosas. Sigue los comandos de arriba y vuelve a ejecutar:")
        print("  python init_entorno.py --todo")
        return 1
    print("\nEntorno listo. Arranca con:\n  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
