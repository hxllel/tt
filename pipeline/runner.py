"""Orquestador: mantiene los modelos cargados una sola vez (RNF-17) y expone las
operaciones que usan la interfaz y los scripts de linea de comandos."""

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline import detect_track, describe, embed, search, superres
from pipeline.config import dir_video, load_config, ruta, _deep_merge
from pipeline.hardware import choose_profile, detect

MODOS = ("clip", "hibrido", "texto", "lexico")


class Motor:
    def __init__(self, cfg: dict = None, overrides: dict = None):
        self.cfg = cfg or load_config(overrides=overrides)
        self._configurar()

    # ------------------------------------------------------------ configuracion
    def _configurar(self):
        import torch
        self.info = detect(self.cfg["dispositivo"])
        self.perfil = choose_profile(self.info, describe.checkpoints_disponibles(self.cfg) or ("0.5b",))
        v = self.cfg["vlm"]
        if v["modelo"] != "auto":
            self.perfil.vlm = v["modelo"]
        if str(v["batch"]) != "auto":
            self.perfil.vlm_batch = int(v["batch"])
        if self.info.device == "cpu":
            torch.set_num_threads(max(1, self.info.cpu_cores))
        self._sr = self._clip = self._texto = self._desc = None

    def reconfigurar(self, overrides: dict):
        self.cfg = _deep_merge(self.cfg, overrides)
        self._configurar()

    @property
    def device(self) -> str:
        return self.info.device

    @property
    def yolo(self) -> str:
        m = self.cfg["deteccion"]["modelo"]
        return self.perfil.yolo if m == "auto" else m

    def superres_activa(self) -> bool:
        a = str(self.cfg["superres"]["activar"]).lower()
        return self.perfil.superres if a == "auto" else a == "true"

    # ------------------------------------------------------------ modelos (lazy)
    def sr(self):
        if self._sr is None:
            self._sr = superres.SuperRes(ruta(self.cfg["modelos_dir"]) / self.cfg["superres"]["modelo"],
                                         self.device, self.perfil.half)
        return self._sr

    def clip(self):
        if self._clip is None:
            self._clip = embed.ClipEmbedder(self.cfg, self.device, self.perfil.half)
        return self._clip

    def texto(self):
        if self._texto is None:
            self._texto = embed.TextEmbedder(self.cfg, self.device, self.perfil.half)
        return self._texto

    def describer(self):
        if self._desc is None:
            self._desc = describe.Describer(self.cfg, self.device, self.perfil.vlm, self.perfil.vlm_batch)
        return self._desc

    def exportar_detector(self) -> Path:
        return detect_track.exportar_detector(self.cfg, self.device, self.perfil.yolo)

    # ------------------------------------------------------------ operaciones
    def procesar_video(self, video_path, procedencia: dict = None, describir: bool = False, progress=None) -> dict:
        meta = detect_track.process_video(video_path, self.cfg, self.device, self.perfil.yolo,
                                          procedencia, progress)
        out_dir = dir_video(self.cfg, meta["video"])
        if meta["crops"]:
            if self.superres_activa():
                superres.apply_to_video(out_dir, self.cfg, self.sr(), progress)
            embed.embed_video_clip(out_dir, self.clip(), self.perfil.clip_batch, progress)
            if describir:
                describe.describir_video(out_dir, self.describer(), None, progress)
        return self.stats(meta["video"])

    def stats(self, video: str) -> dict:
        p = dir_video(self.cfg, video) / "stats.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def describir(self, video: str, crop_ids=None, progress=None) -> dict:
        return describe.describir_video(dir_video(self.cfg, video), self.describer(), crop_ids, progress)

    def _emb_clip(self, out_dir: Path, n_crops: int) -> np.ndarray:
        p = out_dir / "emb_clip.npy"
        if p.exists():
            emb = np.load(p)
            if len(emb) == n_crops:
                return emb
        return embed.embed_video_clip(out_dir, self.clip(), self.perfil.clip_batch)

    def _scores_texto(self, out_dir: Path, descs: dict, consulta: str, crop_ids: list) -> dict:
        cache = out_dir / "emb_texto.npz"
        ids, vecs = [], None
        if cache.exists():
            z = np.load(cache)
            ids, vecs = list(z["ids"]), z["vecs"]
        por_id = {int(i): vecs[k] for k, i in enumerate(ids)} if vecs is not None else {}
        faltan = [c for c in crop_ids if c not in por_id and str(c) in descs and not descs[str(c)].get("error")]
        if faltan:
            nuevos = self.texto().encode([descs[str(c)]["texto_busqueda"] for c in faltan], "passage")
            por_id.update(zip(faltan, nuevos))
            np.savez(cache, ids=np.array(list(por_id), dtype=np.int64), vecs=np.stack(list(por_id.values())))
        q = self.texto().encode([consulta], "query")[0]
        return {c: float(por_id[c] @ q) for c in crop_ids if c in por_id}

    def buscar(self, video: str, consulta: str, modo: str = "clip", top_k: int = None, progress=None) -> list:
        if modo not in MODOS:
            raise ValueError(f"Modo invalido: {modo}")
        if not consulta or not consulta.strip():
            raise ValueError("Escribe la descripcion de la alerta.")
        top_k = top_k or self.cfg["busqueda"]["top_k"]
        out_dir = dir_video(self.cfg, video)
        meta = detect_track.cargar_metadata(out_dir)
        crops = meta["crops"]
        if not crops:
            return []
        tracks = {t["track_id"]: t for t in meta["tracks"]}
        descs = describe.cargar_descripciones(out_dir)

        if modo == "lexico":
            frases = search.frases_atributo(consulta)
            if not frases:
                raise ValueError("No se reconocio ninguna prenda en la consulta.")
            if not descs:
                raise ValueError("No hay descripciones de FastVLM para este video. Generalas primero o usa el modo clip/hibrido.")
            scores = {int(c): search.score_lexico(frases, d.get("texto_busqueda", ""))
                      for c, d in descs.items() if not d.get("error")}
            scores = {c: s for c, s in scores.items() if s > 0}
        elif modo == "texto":
            if not descs:
                raise ValueError("No hay descripciones de FastVLM para este video. Generalas primero o usa el modo clip/hibrido.")
            scores = self._scores_texto(out_dir, descs, consulta, [int(c) for c in descs])
        else:
            emb = self._emb_clip(out_dir, len(crops))
            s = search.score_clip(emb, self.clip().encode_texts(search.textos_clip(consulta)))
            scores = dict(enumerate(s.tolist()))
            if modo == "hibrido":
                por_track = search.agregar_por_track(crops, scores)
                cand = sorted(por_track.values(), key=lambda x: -x[0])[: self.cfg["busqueda"]["candidatos_hibrido"]]
                ids = [cid for _, cid in cand]
                descs = describe.describir_video(out_dir, self.describer(), ids, progress)
                txt = self._scores_texto(out_dir, descs, consulta, ids)
                c_n = search.minmax([s for s, _ in cand])
                t_n = search.minmax([txt.get(cid, 0.0) for cid in ids])
                scores = {cid: float(0.5 * a + 0.5 * b) for cid, a, b in zip(ids, c_n, t_n)}

        ranking = sorted(search.agregar_por_track(crops, scores).items(), key=lambda kv: -kv[1][0])[:top_k]
        resultados = []
        for pos, (tid, (score, cid)) in enumerate(ranking, 1):
            e = crops[cid]
            d = descs.get(str(cid)) or next((descs[str(x["crop_id"])] for x in crops
                                             if x["track_id"] == tid and str(x["crop_id"]) in descs), {})
            t = tracks.get(tid, {})
            resultados.append({
                "posicion": pos, "track_id": tid, "score": round(score, 4), "crop_id": cid,
                "imagen": str(describe.ruta_recorte(out_dir, e)), "timestamp_sec": e["timestamp_sec"],
                "t_inicio": t.get("t_inicio"), "t_fin": t.get("t_fin"), "frame_idx": e["frame_idx"],
                "bbox_yolo": e["bbox_yolo"], "descripcion": d.get("descripcion", ""),
            })
        return resultados

    # ------------------------------------------------------------ visualizacion
    def escena(self, video: str, crop_id: int):
        """Frame completo con la caja de la persona (RGB)."""
        from pipeline.video_io import read_frame
        meta = detect_track.cargar_metadata(dir_video(self.cfg, video))
        e = meta["crops"][crop_id]
        frame = read_frame(meta["video_path"], e["frame_idx"])
        if frame is None:
            return None
        x1, y1, x2, y2 = e["bbox_yolo"]
        grosor = max(2, frame.shape[1] // 400)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), grosor)
        cv2.putText(frame, f"track {e['track_id']}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    grosor * 0.4, (0, 0, 255), grosor)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def tira(self, video: str, crop_id: int, segundos: float = 3.0, n: int = 6) -> list:
        """Frames alrededor de la aparicion (+-segundos) para contexto."""
        from pipeline.video_io import fmt_tiempo, read_frame
        meta = detect_track.cargar_metadata(dir_video(self.cfg, video))
        e = meta["crops"][crop_id]
        fps = meta["fps"]
        salida = []
        for off in np.linspace(-segundos, segundos, n):
            idx = int(max(0, min(meta["frames"] - 1, e["frame_idx"] + off * fps)))
            f = read_frame(meta["video_path"], idx)
            if f is not None:
                f = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
                salida.append((cv2.cvtColor(f, cv2.COLOR_BGR2RGB), fmt_tiempo(idx / fps)))
        return salida
