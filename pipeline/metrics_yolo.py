"""Metricas de deteccion y tracking de personas (Capa 1 del Plan de Validacion).

  (a) Con dataset anotado en formato YOLO (clase 0 = persona):
        python -m pipeline.metrics_yolo dataset --data datos/personas.yaml
  (b) Con ground truth MOT (frame,id,x,y,w,h,...):
        python -m pipeline.metrics_yolo mot --video videos/clip.mp4 --gt gt.txt
  (c) Sin anotaciones (solo estadisticas, NO es exactitud):
        python -m pipeline.metrics_yolo stats --video Video_51
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from pipeline.config import dir_video, load_config, ruta
from pipeline.detect_track import cargar_detector, resolver_detector, yolo_kwargs
from pipeline.video_io import iter_frames

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _figura(nombre: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas xyxy: a (N,4), b (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_greedy(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, iou_thr=0.5):
    """Emparejamiento voraz por confianza. Devuelve (gt_emparejado bool[N], tp, fp)."""
    ious = iou_matrix(gt, pred)
    usado = np.zeros(len(gt), dtype=bool)
    tp = 0
    for j in np.argsort(-conf):
        if len(gt) == 0:
            break
        cand = np.where(~usado, ious[:, j], -1)
        i = int(cand.argmax())
        if cand[i] >= iou_thr:
            usado[i] = True
            tp += 1
    return usado, tp, len(pred) - tp


# ---------------------------------------------------------------- (a) dataset
def _imagenes_val(data_yaml: Path, split: str) -> list:
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(d.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entrada = d.get(split) or d.get("val")
    entradas = entrada if isinstance(entrada, list) else [entrada]
    imgs = []
    for e in entradas:
        p = base / e
        if p.is_dir():
            imgs += sorted(x for x in p.rglob("*") if x.suffix.lower() in EXT_IMG)
        elif p.suffix == ".txt":
            imgs += [base / l.strip() for l in p.read_text().splitlines() if l.strip()]
    return imgs


def _label_de(img: Path) -> Path:
    partes = list(img.parts)
    idx = len(partes) - 1 - partes[::-1].index("images")
    partes[idx] = "labels"
    return Path(*partes).with_suffix(".txt")


def evaluar_dataset(cfg: dict, device: str, perfil_yolo: str, data_yaml, conf_operacion: float = None,
                    split: str = "val", gt_class: int = 0) -> dict:
    data_yaml = Path(data_yaml)
    det = cfg["deteccion"]
    conf_op = conf_operacion if conf_operacion is not None else det["conf_recorte"]
    model_path = resolver_detector(cfg, device, perfil_yolo)
    model = cargar_detector(model_path)
    kw = dict(verbose=False, **yolo_kwargs(cfg, device, model_path))

    # Metricas estandar de ultralytics (P, R, mAP) restringidas a la clase persona
    val = model.val(data=str(data_yaml), split=split, classes=[gt_class], conf=0.001, plots=False, **kw)
    res = {"detector": model_path.name, "precision_ultralytics": round(float(val.box.mp), 4),
           "recall_ultralytics": round(float(val.box.mr), 4), "mAP50": round(float(val.box.map50), 4),
           "mAP50_95": round(float(val.box.map), 4)}

    # Metricas de conteo al umbral de operacion del sistema
    imgs = _imagenes_val(data_yaml, split)
    filas, tam = [], {"chica": [0, 0], "mediana": [0, 0], "grande": [0, 0]}
    tp_t = fp_t = gt_t = 0
    for img, r in zip(imgs, model.predict(source=[str(p) for p in imgs], stream=True, classes=[0],
                                           conf=conf_op, **kw)):
        h, w = r.orig_shape
        gt = []
        lab = _label_de(img)
        if lab.exists():
            for l in lab.read_text().splitlines():
                v = l.split()
                if len(v) >= 5 and int(float(v[0])) == gt_class:
                    cx, cy, bw, bh = (float(x) for x in v[1:5])
                    gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt = np.array(gt, dtype=np.float32).reshape(-1, 4)
        pred = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        emparejado, tp, fp = match_greedy(gt, pred, conf)
        for box, ok in zip(gt, emparejado):
            area = (box[2] - box[0]) * (box[3] - box[1])
            k = "chica" if area < 32 ** 2 else "mediana" if area < 96 ** 2 else "grande"
            tam[k][0] += int(ok)
            tam[k][1] += 1
        tp_t, fp_t, gt_t = tp_t + tp, fp_t + fp, gt_t + len(gt)
        filas.append({"imagen": img.name, "personas_gt": len(gt), "detectadas": len(pred), "tp": tp, "fp": fp,
                      "fn": len(gt) - tp})

    n = max(1, len(filas))
    res.update({
        "conf_operacion": conf_op, "imagenes": len(filas), "personas_gt": gt_t,
        "personas_detectadas_pct": round(100 * tp_t / max(1, gt_t), 2),
        "precision_operacion": round(tp_t / max(1, tp_t + fp_t), 4),
        "falsos_positivos_por_imagen": round(fp_t / n, 3),
        "error_conteo_MAE": round(sum(abs(f["detectadas"] - f["personas_gt"]) for f in filas) / n, 3),
        "recall_por_tamano": {k: (round(v[0] / v[1], 4) if v[1] else None) for k, v in tam.items()},
    })
    out = ruta(cfg["salidas_dir"]) / "metricas"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"dataset_{data_yaml.stem}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0]) if filas else ["imagen"])
        wr.writeheader()
        wr.writerows(filas)
    (out / f"dataset_{data_yaml.stem}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- (b) MOT
def evaluar_mot(cfg: dict, device: str, perfil_yolo: str, video_path, gt_path, intervalo: int = 1,
                clase_mot: int = None) -> dict:
    import motmetrics as mm
    det = cfg["deteccion"]
    gt = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    if gt.shape[1] >= 7:
        gt = gt[gt[:, 6] != 0]            # columna "considerar" de MOTChallenge
    if clase_mot is not None and gt.shape[1] >= 8:
        gt = gt[gt[:, 7] == clase_mot]
    por_frame = {}
    for row in gt:
        por_frame.setdefault(int(row[0]), []).append(row)

    model_path = resolver_detector(cfg, device, perfil_yolo)
    model = cargar_detector(model_path)
    model.predictor = None
    acc = mm.MOTAccumulator(auto_id=False)
    for idx, _, frame in iter_frames(video_path, intervalo):
        f_mot = idx + 1
        r = model.track(frame, classes=[0], conf=det["conf_tracker"], tracker=det["tracker"], persist=True,
                        verbose=False, **yolo_kwargs(cfg, device, model_path))[0]
        hyp_ids, hyp = [], np.zeros((0, 4))
        if r.boxes is not None and r.boxes.id is not None:
            keep = r.boxes.conf.cpu().numpy() >= det["conf_recorte"]
            hyp = r.boxes.xywh.cpu().numpy()[keep]
            hyp[:, :2] -= hyp[:, 2:] / 2   # centro -> esquina superior izquierda
            hyp_ids = r.boxes.id.cpu().numpy().astype(int)[keep].tolist()
        filas = por_frame.get(f_mot, [])
        gt_ids = [int(x[1]) for x in filas]
        gt_boxes = np.array([x[2:6] for x in filas]).reshape(-1, 4)
        acc.update(gt_ids, hyp_ids, mm.distances.iou_matrix(gt_boxes, hyp, max_iou=0.5), frameid=f_mot)

    mh = mm.metrics.create()
    nombres = ["num_frames", "mota", "motp", "idf1", "precision", "recall", "num_switches",
               "num_false_positives", "num_misses", "num_objects", "num_unique_objects",
               "mostly_tracked", "mostly_lost"]
    s = mh.compute(acc, metrics=nombres, name="video").iloc[0].to_dict()
    res = {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in s.items()}
    res["motp_nota"] = "motp de motmetrics es 1 - IoU medio (menor es mejor)"
    res["detector"] = model_path.name
    res["intervalo_frames"] = intervalo
    return res


# ---------------------------------------------------------------- (c) sin GT
def estadisticas_sin_gt(cfg: dict, video: str) -> tuple:
    out_dir = dir_video(cfg, video)
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))["deteccion"]
    dpf = np.array(stats["detecciones_por_frame"])
    lt = np.array(stats["longitud_tracks"] or [0])
    resumen = {
        "AVISO": "Sin anotaciones esto NO mide exactitud: solo describe lo que el detector reporta.",
        "frames_muestreados": stats["frames_muestreados"],
        "personas_unicas_guardadas": stats["tracks_guardados"],
        "detecciones_totales": stats["detecciones_totales"],
        "detecciones_por_frame_media": round(float(dpf.mean()), 2) if dpf.size else 0,
        "max_personas_en_frame": stats["max_personas_en_frame"],
        "frames_sin_personas_pct": round(100 * float((dpf == 0).mean()), 2) if dpf.size else 0,
        "frames_baja_luz": stats["frames_baja_luz"],
        "longitud_track_mediana": float(np.median(lt)),
        "fps_procesados": stats["fps_procesados"],
    }
    plt = _figura("stats")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(dpf, lw=0.8)
    ax[0].set_title("Detecciones por frame muestreado")
    ax[0].set_xlabel("frame muestreado")
    ax[1].hist(lt, bins=30)
    ax[1].set_title("Longitud de tracks (detecciones)")
    ax[2].bar(np.linspace(0.025, 0.975, 20), stats["hist_confianza"], width=0.045)
    ax[2].set_title("Histograma de confianza")
    fig.tight_layout()
    png = out_dir / "metricas_sin_gt.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return resumen, png


def main():
    from pipeline.hardware import choose_profile, detect
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--dispositivo", default=None)
    sub = ap.add_subparsers(dest="modo", required=True)
    a = sub.add_parser("dataset")
    a.add_argument("--data", required=True)
    a.add_argument("--conf", type=float)
    a.add_argument("--split", default="val")
    b = sub.add_parser("mot")
    b.add_argument("--video", required=True)
    b.add_argument("--gt", required=True)
    b.add_argument("--intervalo", type=int, default=1)
    b.add_argument("--clase", type=int)
    c = sub.add_parser("stats")
    c.add_argument("--video", required=True, help="nombre del video ya procesado en salidas/")
    args = ap.parse_args()

    cfg = load_config(args.config, {"dispositivo": args.dispositivo} if args.dispositivo else None)
    if args.modo == "stats":
        res, png = estadisticas_sin_gt(cfg, args.video)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"Grafica: {png}")
        return
    info = detect(cfg["dispositivo"])
    yolo = choose_profile(info).yolo
    if args.modo == "dataset":
        res = evaluar_dataset(cfg, info.device, yolo, args.data, args.conf, args.split)
    else:
        res = evaluar_mot(cfg, info.device, yolo, args.video, args.gt, args.intervalo, args.clase)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
