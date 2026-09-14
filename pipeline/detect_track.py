"""Etapa 1: deteccion (YOLO26) + tracking (ByteTrack) con seleccion de los mejores
recortes por persona.

Optimizaciones frente a la version original:
  - una sola copia GPU->CPU por frame (no .item() por caja)
  - conf baja para el tracker (ByteTrack asocia con detecciones debiles)
  - solo se guardan los top-K recortes de cada track, escritos en un hilo aparte
    cuando el track se cierra (memoria acotada con miles de personas)
  - CLAHE solo en frames oscuros (RF-03)
  - motor TensorRT (.engine) / OpenVINO si existe
"""

import heapq
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.config import dir_video, ruta
from pipeline.video_io import iter_frames, video_props

_MODELOS = {}


# ---------------------------------------------------------------- utilidades
def expand_bbox(xmin, ymin, xmax, ymax, img_w, img_h, margin=0.10):
    pad_x = int(round((xmax - xmin) * margin))
    pad_y = int(round((ymax - ymin) * margin))
    return (max(0, xmin - pad_x), max(0, ymin - pad_y),
            min(img_w, xmax + pad_x), min(img_h, ymax + pad_y))


def quality_score(crop, bbox, img_w, img_h, conf) -> float:
    """Calidad del recorte en [0, 1]: confianza x tamano x nitidez x no truncado."""
    xmin, ymin, xmax, ymax = bbox
    h = ymax - ymin
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > 128:
        gray = cv2.resize(gray, (max(1, gray.shape[1] * 128 // gray.shape[0]), 128))
    nitidez = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 200.0)
    toca_borde = xmin <= 1 or ymin <= 1 or xmax >= img_w - 1 or ymax >= img_h - 1
    return float(conf) * min(1.0, h / 200.0) * (0.5 + 0.5 * nitidez) * (0.6 if toca_borde else 1.0)


def mejorar_baja_luz(frame, umbral: float):
    """Aplica CLAHE al canal L si la luminancia media es menor al umbral."""
    small = cv2.resize(frame, (160, 90))
    if cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean() >= umbral:
        return frame, False
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), True


@dataclass
class TrackState:
    track_id: int
    k: int
    first_t: float = 0.0
    last_t: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    last_call: int = 0
    n_det: int = 0
    heap: list = field(default_factory=list)   # (quality, frame_idx, entry, crop)

    def add(self, quality, frame_idx, t, call_idx, entry, crop):
        if self.n_det == 0:
            self.first_t, self.first_frame = t, frame_idx
        self.n_det += 1
        self.last_t, self.last_frame, self.last_call = t, frame_idx, call_idx
        if crop is None:
            return
        item = (quality, frame_idx, entry, crop)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif quality > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def best(self):
        return sorted(self.heap, key=lambda x: (-x[0], x[1]))


# ---------------------------------------------------------------- modelo
def yolo_kwargs(cfg: dict, hw_device: str, model_path: Path) -> dict:
    """Dispositivo, imgsz y precision para predict/track/val (ultralytics 8.4: quantize reemplaza half)."""
    imgsz = cfg["deteccion"]["imgsz"]
    if str(imgsz) == "auto":
        imgsz = 960 if hw_device == "cuda" else 640
    kw = {"device": 0 if hw_device == "cuda" else "cpu", "imgsz": int(imgsz)}
    if hw_device == "cuda" and Path(model_path).suffix == ".pt":
        kw["quantize"] = 16
    return kw


def resolver_detector(cfg: dict, hw_device: str, perfil_yolo: str) -> Path:
    det = cfg["deteccion"]
    nombre = perfil_yolo if det["modelo"] == "auto" else det["modelo"]
    modelos = ruta(cfg["modelos_dir"])
    pt = modelos / f"{nombre}.pt"
    motor = det["motor"]
    engine = modelos / f"{nombre}.engine"
    openvino = modelos / f"{nombre}_openvino_model"
    if motor in ("auto", "engine") and hw_device == "cuda" and engine.exists():
        return engine
    if motor in ("auto", "openvino") and hw_device == "cpu" and openvino.exists():
        return openvino
    return pt


def cargar_detector(path: Path):
    from ultralytics import YOLO
    key = str(path)
    if key not in _MODELOS:
        _MODELOS[key] = YOLO(key, task="detect")
    return _MODELOS[key]


def exportar_detector(cfg: dict, hw_device: str, perfil_yolo: str) -> Path:
    """Exporta el .pt a TensorRT (GPU) u OpenVINO (CPU)."""
    from ultralytics import YOLO
    det = cfg["deteccion"]
    nombre = perfil_yolo if det["modelo"] == "auto" else det["modelo"]
    pt = ruta(cfg["modelos_dir"]) / f"{nombre}.pt"
    model = YOLO(str(pt))
    if hw_device == "cuda":
        return Path(model.export(format="engine", quantize=16, imgsz=yolo_kwargs(cfg, hw_device, pt)["imgsz"], device=0))
    return Path(model.export(format="openvino", imgsz=yolo_kwargs(cfg, hw_device, pt)["imgsz"]))


# ---------------------------------------------------------------- proceso
def process_video(video_path, cfg: dict, hw_device: str, perfil_yolo: str,
                  procedencia: dict = None, progress=None) -> dict:
    """Procesa un video y escribe salidas/<video>/{metadata.json, stats.json, crops/}."""
    det = cfg["deteccion"]
    video_path = Path(video_path)
    video_name = video_path.stem
    out_dir = dir_video(cfg, video_name)
    if out_dir.exists():  # reprocesar invalida recortes, embeddings y descripciones previos
        shutil.rmtree(out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    props = video_props(video_path)
    model_path = resolver_detector(cfg, hw_device, perfil_yolo)
    model = cargar_detector(model_path)
    track_kwargs = dict(
        classes=[0], conf=det["conf_tracker"], tracker=det["tracker"], persist=True, verbose=False,
        **yolo_kwargs(cfg, hw_device, model_path),
    )

    activos: dict = {}
    crops_meta = []
    tracks_meta = {}
    det_por_frame = []
    conf_hist = np.zeros(20, dtype=np.int64)
    frames_baja_luz = 0
    writer = ThreadPoolExecutor(max_workers=2)
    pendientes = []
    t0 = time.perf_counter()
    total_muestreados = max(1, props["frames"] // det["intervalo_frames"])

    def cerrar(ts: TrackState):
        if ts.n_det < det["min_detecciones_track"] or not ts.heap:
            return
        tracks_meta[ts.track_id] = {
            "track_id": ts.track_id, "n_detecciones": ts.n_det,
            "t_inicio": ts.first_t, "t_fin": ts.last_t,
            "frame_inicio": ts.first_frame, "frame_fin": ts.last_frame,
        }
        for rank, (q, fidx, entry, crop) in enumerate(ts.best()):
            nombre = f"track_{ts.track_id:05d}_r{rank}_f{fidx}.jpg"
            entry = {**entry, "rank": rank, "crop_path": f"crops/{nombre}"}
            crops_meta.append(entry)
            pendientes.append(writer.submit(cv2.imwrite, str(crops_dir / nombre), crop))

    # Reinicia el estado del tracker entre videos (persist=True lo conserva)
    if getattr(model, "predictor", None) is not None:
        model.predictor = None

    call_idx = 0
    for frame_idx, t, frame in iter_frames(video_path, det["intervalo_frames"]):
        call_idx += 1
        frame_in, oscuro = mejorar_baja_luz(frame, det["umbral_baja_luz"])
        frames_baja_luz += int(oscuro)
        r = model.track(frame_in, **track_kwargs)[0]
        boxes = r.boxes
        n = 0 if boxes is None else len(boxes)
        det_por_frame.append(n)
        if n and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            conf_hist += np.histogram(confs, bins=20, range=(0, 1))[0]
            img_h, img_w = frame_in.shape[:2]
            for (xmin, ymin, xmax, ymax), tid, conf in zip(xyxy, ids, confs):
                ts = activos.get(tid)
                if ts is None:
                    ts = activos[tid] = TrackState(int(tid), det["top_k_por_track"])
                crop, entry, q = None, None, 0.0
                if conf >= det["conf_recorte"] and (ymax - ymin) >= det["min_alto_px"]:
                    ex = expand_bbox(xmin, ymin, xmax, ymax, img_w, img_h, det["margen_contexto"])
                    crop = frame_in[ex[1]:ex[3], ex[0]:ex[2]]
                    if crop.size:
                        q = quality_score(crop, (xmin, ymin, xmax, ymax), img_w, img_h, conf)
                        # Copia solo si entra al top-K (evita copiar miles de recortes)
                        if len(ts.heap) < ts.k or q > ts.heap[0][0]:
                            crop = crop.copy()
                            entry = {
                                "video": video_name, "track_id": int(tid), "frame_idx": int(frame_idx),
                                "timestamp_sec": t, "bbox_yolo": [int(xmin), int(ymin), int(xmax), int(ymax)],
                                "bbox_con_contexto": list(map(int, ex)), "confidence": round(float(conf), 4),
                                "calidad": round(q, 4), "baja_luz": oscuro,
                            }
                        else:
                            crop = None
                    else:
                        crop = None
                ts.add(q, int(frame_idx), t, call_idx, entry, crop)

        # Cierra tracks que no se han visto en un rato (libera memoria)
        viejos = [tid for tid, ts in activos.items()
                  if call_idx - ts.last_call > det["frames_para_cerrar_track"]]
        for tid in viejos:
            cerrar(activos.pop(tid))

        if progress and call_idx % 10 == 0:
            progress(min(0.99, call_idx / total_muestreados),
                     f"Deteccion: frame {frame_idx}/{props['frames']} | personas {len(tracks_meta) + len(activos)}")

    for ts in activos.values():
        cerrar(ts)
    for p in pendientes:
        p.result()
    writer.shutdown()

    dur = time.perf_counter() - t0
    crops_meta.sort(key=lambda e: (e["track_id"], e["rank"]))
    for i, e in enumerate(crops_meta):
        e["crop_id"] = i
    longitudes = [m["n_detecciones"] for m in tracks_meta.values()]

    metadata = {
        "video": video_name, "video_path": str(video_path.resolve()), **props,
        "intervalo_frames": det["intervalo_frames"], "detector": str(model_path.name),
        "dispositivo": hw_device, "procedencia": procedencia or {},
        "tracks": list(tracks_meta.values()), "crops": crops_meta,
    }
    stats = {
        "frames_muestreados": len(det_por_frame), "frames_baja_luz": frames_baja_luz,
        "detecciones_por_frame": det_por_frame,
        "detecciones_totales": int(sum(det_por_frame)),
        "max_personas_en_frame": int(max(det_por_frame, default=0)),
        "tracks_guardados": len(tracks_meta), "recortes": len(crops_meta),
        "longitud_tracks": longitudes, "hist_confianza": conf_hist.tolist(),
        "tiempo_deteccion_s": round(dur, 2),
        "fps_procesados": round(len(det_por_frame) / dur, 2) if dur else 0,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=1)
    _actualizar_stats(out_dir, {"deteccion": stats})
    return metadata


def _actualizar_stats(out_dir: Path, nuevos: dict):
    path = out_dir / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stats.update(nuevos)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")


def cargar_metadata(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / "metadata.json").read_text(encoding="utf-8"))
