"""Lectura eficiente de video: grab() en los frames saltados (sin conversion de
color ni copia) y retrieve() solo en los muestreados, en un hilo aparte."""

import queue
import threading

import cv2


def video_props(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {path}")
    props = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return props


def iter_frames(path, interval: int, queue_size: int = 8):
    """Genera (frame_idx, timestamp_sec, frame_bgr) para 1 de cada `interval` frames.

    frame_idx es 0-based. El timestamp sale de CAP_PROP_POS_MSEC (correcto con
    FPS variable, comun en .mov de celular) con respaldo en idx/fps."""
    q = queue.Queue(maxsize=queue_size)
    stop = threading.Event()
    FIN = object()

    def lector():
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = -1
        try:
            while not stop.is_set():
                if not cap.grab():
                    break
                idx += 1
                if idx % interval:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if t <= 0 and idx > 0:
                    t = idx / fps
                q.put((idx, round(t, 3), frame))
        finally:
            cap.release()
            q.put(FIN)

    hilo = threading.Thread(target=lector, daemon=True)
    hilo.start()
    try:
        while True:
            item = q.get()
            if item is FIN:
                break
            yield item
    finally:
        stop.set()
        # Vacia la cola para que el hilo no quede bloqueado en put()
        while hilo.is_alive():
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


def read_frame(path, frame_idx: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def fmt_tiempo(seg: float) -> str:
    m, s = divmod(max(0.0, seg), 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"
