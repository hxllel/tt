"""
Etapa 2: Busqueda con FastVLM sobre los recortes ya extraidos
----------------------------------------------------------------
Lee el metadata.json generado por extract_bounding_boxes.py (que
organiza los recortes en una carpeta por PERSONA UNICA -track_id-
en vez de una carpeta por frame) y manda UN SOLO recorte
representativo por persona (por defecto el de mayor confianza de
deteccion) a FastVLM para obtener su descripcion, en vez de
analizar cada frame de cada persona. Guarda todo en
resultados_busqueda.json y, opcionalmente, filtra coincidencias
por palabras clave.

Nota sobre rutas: el metadata.json ya trae el "crop_path" absoluto
de cada recorte (dentro de su carpeta persona_{track_id}_video_{nombre}),
asi que este script no necesita reconstruir rutas por frame; solo
lee crop_path tal cual quedo guardado en metadata.json.

Uso:
    python3 search_fastvlm.py
(edita la config abajo, y el nombre de video en el bloque main)
"""

import subprocess
import os
import json

PYTHON_EXEC = "/home/hillel/anaconda3/envs/fastvlm/bin/python3"
PREDICT_SCRIPT = "/home/hillel/Escritorio/tt/1/ml-fastvlm/predict.py"
MODEL_PATH = "/home/hillel/Escritorio/tt/1/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage2/llava-fastvithd_0.5b_stage2"

# Apunta a la carpeta que quieras analizar: la de recortes originales
# o la de super-resolucion (bounding_boxes_sr_hat), ambas comparten
# la misma estructura por persona y el mismo metadata.json.
BOUNDING_BOXES_ROOT = "/home/hillel/Escritorio/tt/bounding_boxes"


PROMPT = "Describe a la persona en la imagen en español: color de playera, color de pantalon y cualquier detalle distintivo."
# -------------------------------------------------


def run_fastvlm(image_path: str, prompt: str = PROMPT) -> str:
    command = [
        PYTHON_EXEC, PREDICT_SCRIPT,
        "--model-path", MODEL_PATH,
        "--image-file", image_path,
        "--prompt", prompt,
    ]
    proceso = subprocess.run(command, capture_output=True, text=True)
    if proceso.stderr:
        print(f"  STDERR ({image_path}): {proceso.stderr.strip()}")
    return proceso.stdout.strip()


def seleccionar_representativos(metadata, criterio="confidence"):
    """
    Agrupa las entradas de metadata por carpeta de persona (track_id) y
    elige UNA sola por grupo, para no mandar cada frame de la misma
    persona a FastVLM.

    criterio:
        "confidence" -> elige el recorte con mayor confianza de deteccion
        "primero"    -> elige simplemente el primer frame en que aparecio
    """
    por_persona = {}
    for entry in metadata:
        track_id = entry["track_id"]
        por_persona.setdefault(track_id, []).append(entry)

    representativos = []
    for track_id, entradas in por_persona.items():
        if criterio == "confidence":
            elegido = max(entradas, key=lambda e: e.get("confidence", 0))
        else:  # "primero"
            elegido = min(entradas, key=lambda e: e["frame_id"])
        representativos.append(elegido)

    return representativos


def search_video(video_name: str, bounding_boxes_root: str = BOUNDING_BOXES_ROOT,
                  prompt: str = PROMPT, un_recorte_por_persona: bool = True,
                  criterio_representativo: str = "confidence"):
    video_dir = os.path.join(bounding_boxes_root, video_name)
    metadata_path = os.path.join(video_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No existe {metadata_path}. Corre primero extract_bounding_boxes.py"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if un_recorte_por_persona:
        entradas_a_analizar = seleccionar_representativos(metadata, criterio_representativo)
        print(f"Analizando 1 recorte por persona ({len(entradas_a_analizar)} "
              f"personas unicas de {len(metadata)} recortes totales) de "
              f"'{video_name}' con FastVLM...")
    else:
        entradas_a_analizar = metadata
        print(f"Analizando {len(metadata)} recortes de '{video_name}' con FastVLM...")

    resultados = []

    for entry in entradas_a_analizar:
        crop_path = entry["crop_path"]
        if not os.path.exists(crop_path):
            print(f"  [aviso] no existe {crop_path}, se omite "
                  f"(persona {entry.get('track_id')}, frame {entry.get('frame_id')})")
            continue

        descripcion = run_fastvlm(crop_path, prompt)
        resultados.append({**entry, "descripcion": descripcion})

        print(f"[persona {entry['track_id']} | frame {entry['frame_id']} | "
              f"t={entry['timestamp_sec']}s] {descripcion}")

    results_path = os.path.join(video_dir, "resultados_busqueda.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"\nResultados guardados en: {results_path}")
    return resultados


def find_matches(resultados, palabras_clave):
    """Filtra resultados cuya descripcion contenga TODAS las palabras clave dadas."""
    palabras_clave = [p.lower() for p in palabras_clave]
    return [
        r for r in resultados
        if all(p in r["descripcion"].lower() for p in palabras_clave)
    ]


def resumen_por_persona(resultados):
    """Agrupa las descripciones por persona (track_id) para revisar de
    un vistazo que dijo FastVLM sobre cada persona unica a lo largo
    del video."""
    por_persona = {}
    for r in resultados:
        por_persona.setdefault(r["track_id"], []).append(r)
    return por_persona


if __name__ == "__main__":
    resultados = search_video("video1")

    # Ejemplo de filtrado posterior por palabras clave:
    coincidencias = find_matches(resultados, ["playera azul", "mezclilla"])
    print(f"\n{len(coincidencias)} coincidencia(s) encontradas:")
    for c in coincidencias:
        print(f"  Persona {c['track_id']} | Frame {c['frame_id']} "
              f"(t={c['timestamp_sec']}s): {c['descripcion']}")