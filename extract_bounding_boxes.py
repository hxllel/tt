"""
Etapa 1: Extraccion de bounding boxes con YOLO26 + Tracking
--------------------------------------------------------------
Recorre uno o varios videos, detecta y trackea personas cada N
frames, recorta cada deteccion (expandiendo el bbox de YOLO un
10% en cada direccion para dar contexto, con clamp a los limites
del frame) y la guarda organizada en subcarpetas por video y por
ID UNICO de persona (track_id), junto con un metadata.json que
enlaza cada recorte con su frame, timestamp, ambos bboxes y
track_id.

Uso:
    python3 extract_bounding_boxes.py
(edita VIDEO_PATHS y OUTPUT_ROOT abajo antes de correr)
"""

import cv2
import os
import json
from ultralytics import YOLO

# ---------------- CONFIGURACION ----------------
VIDEO_PATHS = [
    "/home/hillel/Escritorio/tt/video1.mp4",
]
OUTPUT_ROOT = "/home/hillel/Escritorio/tt/bounding_boxes"
FRAME_INTERVAL = 10        # procesa 1 de cada N frames
YOLO_MODEL_PATH = "yolo26n.pt"
CONF_THRESHOLD = 0.5
TRACKER_CFG = "bytetrack.yaml"   # tracker de ultralytics (bytetrack.yaml / botsort.yaml)
CONTEXT_MARGIN = 0.10       
# -------------------------------------------------


def get_video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def expand_bbox_with_context(xmin, ymin, xmax, ymax, img_w, img_h, margin=CONTEXT_MARGIN):
    """
    Expande un bbox un `margin` (proporcion, ej. 0.10 = 10%) del ancho y
    alto de la propia caja, hacia los 4 lados, y recorta (clamp) el
    resultado a los limites del frame.
    """
    box_w = xmax - xmin
    box_h = ymax - ymin

    pad_x = int(round(box_w * margin))
    pad_y = int(round(box_h * margin))

    new_xmin = max(0, xmin - pad_x)
    new_ymin = max(0, ymin - pad_y)
    new_xmax = min(img_w, xmax + pad_x)
    new_ymax = min(img_h, ymax + pad_y)

    return new_xmin, new_ymin, new_xmax, new_ymax


def extract_bounding_boxes(video_path, model, output_root=OUTPUT_ROOT,
                            frame_interval=FRAME_INTERVAL, conf=CONF_THRESHOLD,
                            tracker_cfg=TRACKER_CFG):
    video_name = get_video_name(video_path)
    video_output_dir = os.path.join(output_root, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    metadata = []

    # Cuenta cuantos recortes lleva cada track_id, para no repetir
    # nombres de archivo dentro de la carpeta de esa persona.
    track_crop_counts = {}

    print(f"\nProcesando '{video_name}' (1 de cada {frame_interval} frames, con tracking)...")

    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % frame_interval != 0:
            continue

        # IMPORTANTE: persist=True mantiene el estado del tracker entre
        # llamadas para que los track_id se conserven a lo largo del video,
        # incluso saltando frames con FRAME_INTERVAL.
        results = model.track(
            source=img,
            classes=[0],
            conf=conf,
            tracker=tracker_cfg,
            persist=True,
            verbose=False,
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            if result.boxes.id is None:
                # El tracker aun no asigno IDs a estas detecciones (frame
                # inicial o deteccion perdida); se descartan para evitar
                # personas "fantasma" sin identidad.
                continue

            img_h, img_w = img.shape[:2]

            for box in result.boxes:
                track_id = int(box.id[0].item())
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                # Bbox original de YOLO + version expandida con contexto,
                # esta ultima es la que realmente se recorta y guarda.
                cxmin, cymin, cxmax, cymax = expand_bbox_with_context(
                    xmin, ymin, xmax, ymax, img_w, img_h
                )

                persona = img[cymin:cymax, cxmin:cxmax]
                if persona.size == 0:
                    continue

                # Una carpeta por persona unica (track_id), no por frame.
                person_dir = os.path.join(
                    video_output_dir,
                    f"persona_{track_id}_video_{video_name}"
                )
                os.makedirs(person_dir, exist_ok=True)

                crop_idx = track_crop_counts.get(track_id, 0)
                track_crop_counts[track_id] = crop_idx + 1

                crop_filename = f"frame_{frame_id}_crop_{crop_idx}.jpg"
                crop_path = os.path.join(person_dir, crop_filename)
                cv2.imwrite(crop_path, persona)

                metadata.append({
                    "video": video_name,
                    "frame_id": frame_id,
                    "timestamp_sec": round(frame_id / fps, 2),
                    "track_id": track_id,
                    "bbox_yolo": [xmin, ymin, xmax, ymax],
                    "bbox_con_contexto": [cxmin, cymin, cxmax, cymax],
                    "confidence": float(box.conf[0]),
                    "crop_path": crop_path,
                })

        if frame_id % 300 == 0:
            print(f"  ... frame {frame_id}")

    cap.release()

    n_personas_unicas = len(track_crop_counts)
    metadata_path = os.path.join(video_output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"'{video_name}': {len(metadata)} recortes, {n_personas_unicas} personas unicas -> {video_output_dir}")
    return metadata_path


def main():
    model = YOLO(YOLO_MODEL_PATH)  # se carga una sola vez para todos los videos
    resumen = {}
    for video_path in VIDEO_PATHS:
        video_name = get_video_name(video_path)
        resumen[video_name] = extract_bounding_boxes(video_path, model)

    print("\nExtraccion completa.")
    for video_name, path in resumen.items():
        print(f"  {video_name}: {path}")


if __name__ == "__main__":
    main()