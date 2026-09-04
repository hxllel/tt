"""
Etapa 1: Extraccion de bounding boxes con YOLO26
--------------------------------------------------
Recorre uno o varios videos, detecta personas cada N frames,
recorta cada deteccion y la guarda organizada en subcarpetas
por video y por frame, junto con un metadata.json que enlaza
cada recorte con su frame, timestamp y bbox.

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
    "/home/hillel/Escritorio/tt/Video_51.mov",
    
]
OUTPUT_ROOT = "/home/hillel/Escritorio/tt/bounding_boxes"
FRAME_INTERVAL = 10        # procesa 1 de cada N frames
YOLO_MODEL_PATH = "yolo26n.pt"
CONF_THRESHOLD = 0.5
# -------------------------------------------------


def get_video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def extract_bounding_boxes(video_path, model, output_root=OUTPUT_ROOT,
                            frame_interval=FRAME_INTERVAL, conf=CONF_THRESHOLD):
    video_name = get_video_name(video_path)
    video_output_dir = os.path.join(output_root, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    metadata = []

    print(f"\nProcesando '{video_name}' (1 de cada {frame_interval} frames)...")

    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % frame_interval != 0:
            continue

        results = model.predict(source=img, classes=[0], conf=conf, verbose=False)

        for result in results:
            if len(result.boxes) == 0:
                continue

            frame_dir = os.path.join(
                video_output_dir,
                f"bounding_boxes_frame_{frame_id}_video_{video_name}"
            )
            os.makedirs(frame_dir, exist_ok=True)

            for i, box in enumerate(result.boxes):
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                persona = img[ymin:ymax, xmin:xmax]
                if persona.size == 0:
                    continue

                crop_path = os.path.join(frame_dir, f"persona_{i}.jpg")
                cv2.imwrite(crop_path, persona)

                metadata.append({
                    "video": video_name,
                    "frame_id": frame_id,
                    "timestamp_sec": round(frame_id / fps, 2),
                    "persona_idx": i,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": float(box.conf[0]),
                    "crop_path": crop_path,
                })

        if frame_id % 300 == 0:
            print(f"  ... frame {frame_id}")

    cap.release()

    metadata_path = os.path.join(video_output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"'{video_name}': {len(metadata)} recortes -> {video_output_dir}")
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
