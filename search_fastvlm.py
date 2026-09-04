"""
Etapa 2: Busqueda con FastVLM sobre los recortes ya extraidos
----------------------------------------------------------------
Lee el metadata.json generado por extract_bounding_boxes.py y
manda cada recorte a FastVLM (como proceso externo) para obtener
una descripcion. Guarda todo en resultados_busqueda.json y,
opcionalmente, filtra coincidencias por palabras clave.

Uso:
    python3 search_fastvlm.py
(edita la config abajo, y el nombre de video en el bloque main)
"""

import subprocess
import os
import json

# ---------------- CONFIGURACION ----------------
PYTHON_EXEC = "/home/hillel/anaconda3/envs/fastvlm/bin/python3"
PREDICT_SCRIPT = "/home/hillel/Escritorio/tt/1/ml-fastvlm/predict.py"
MODEL_PATH = "/home/hillel/Escritorio/tt/1/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage2/llava-fastvithd_0.5b_stage2"
BOUNDING_BOXES_ROOT = "/home/hillel/Escritorio/tt/bounding_boxes"

# Prompt generico: describe a la persona (permite reutilizar los mismos
# recortes para distintas busquedas sin volver a correr YOLO ni FastVLM
# sobre todo el video).
PROMPT = "Describe a la persona en la imagen en espanol: color de playera, color de pantalon y cualquier detalle distintivo."
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


def search_video(video_name: str, bounding_boxes_root: str = BOUNDING_BOXES_ROOT,
                  prompt: str = PROMPT):
    video_dir = os.path.join(bounding_boxes_root, video_name)
    metadata_path = os.path.join(video_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No existe {metadata_path}. Corre primero extract_bounding_boxes.py"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    resultados = []
    print(f"Analizando {len(metadata)} recortes de '{video_name}' con FastVLM...")

    for entry in metadata:
        crop_path = entry["crop_path"]
        if not os.path.exists(crop_path):
            continue

        descripcion = run_fastvlm(crop_path, prompt)
        resultados.append({**entry, "descripcion": descripcion})

        print(f"[frame {entry['frame_id']} | t={entry['timestamp_sec']}s | "
              f"persona {entry['persona_idx']}] {descripcion}")

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


if __name__ == "__main__":
    resultados = search_video("Video_51")

    # Ejemplo de filtrado posterior por palabras clave:
    coincidencias = find_matches(resultados, ["playera azul", "mezclilla"])
    print(f"\n{len(coincidencias)} coincidencia(s) encontradas:")
    for c in coincidencias:
        print(f"  Frame {c['frame_id']} (t={c['timestamp_sec']}s): {c['descripcion']}")
