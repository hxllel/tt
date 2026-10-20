"""Orquestador: mantiene los modelos cargados una sola vez (RNF-17) y expone las
operaciones que usan la interfaz y los scripts de linea de comandos.

Toda la lectura de la imagen la hace FastVLM: describe cada persona una sola vez
y la busqueda trabaja sobre ese texto, por similitud semantica (e5) o por
coincidencia lexica de atributos."""

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline import detect_track, describe, embed, search, superres
from pipeline.config import dir_video, load_config, ruta, _deep_merge
from pipeline.hardware import choose_profile, detect

MODOS = ("combinado", "texto", "lexico")
SIN_DESCRIPCIONES = ("Este video no tiene descripciones de FastVLM. Generalas con "
                     "'Generar descripciones FastVLM de todo el video' "
                     "(o `python search_fastvlm.py <video> --describir`).")


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
        self._sr = self._texto = self._desc = None

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
    def procesar_video(self, video_path, procedencia: dict = None, describir: bool = True,
                       progress=None) -> dict:
        """Detecta y trackea, mejora los recortes pequenos y (por defecto) genera
        la descripcion FastVLM de cada persona, que es lo que hace buscable el video."""
        meta = detect_track.process_video(video_path, self.cfg, self.device, self.perfil.yolo,
                                          procedencia, progress)
        out_dir = dir_video(self.cfg, meta["video"])
        if meta["crops"]:
            if self.superres_activa():
                superres.apply_to_video(out_dir, self.cfg, self.sr(), progress)
            if describir:
                describe.describir_video(out_dir, self.describer(), None, progress)
        return self.stats(meta["video"])

    def stats(self, video: str) -> dict:
        p = dir_video(self.cfg, video) / "stats.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def describir(self, video: str, crop_ids=None, progress=None) -> dict:
        return describe.describir_video(dir_video(self.cfg, video), self.describer(), crop_ids, progress)

    def _scores_lexico(self, consulta: str, utiles: dict, exigir_prenda: bool) -> dict:
        frases = search.frases_atributo(consulta)
        if not frases:
            if exigir_prenda:
                raise ValueError("No se reconocio ninguna prenda en la consulta "
                                 "(playera, pantalon, mochila...). Revisa la redaccion "
                                 "o usa el modo semantico.")
            return {}
        return {c: search.score_lexico(frases, d.get("texto_busqueda", "")) for c, d in utiles.items()}

    def _scores_texto(self, out_dir: Path, consulta: str, descs: dict, utiles: dict, progress) -> dict:
        vecs = embed.embeddings_descripciones(out_dir, self.texto(), descs, list(utiles), progress)
        # FastVLM describe en ingles: traducir la consulta con el diccionario de
        # atributos mejora bastante el emparejamiento.
        en = search.consulta_en_ingles(consulta)
        q = self.texto().encode([en or consulta], "query")[0]
        return {c: float(v @ q) for c, v in vecs.items()}

    def buscar(self, video: str, consulta: str, modo: str = "combinado", top_k: int = None,
               progress=None) -> list:
        if modo not in MODOS:
            raise ValueError(f"Modo invalido: {modo}. Usa uno de {MODOS}.")
        if not consulta or not consulta.strip():
            raise ValueError("Escribe la descripcion de la alerta.")
        top_k = top_k or self.cfg["busqueda"]["top_k"]
        out_dir = dir_video(self.cfg, video)
        meta = detect_track.cargar_metadata(out_dir)
        crops = meta["crops"]
        if not crops:
            return []
        descs = describe.cargar_descripciones(out_dir)
        if not descs:
            raise ValueError(SIN_DESCRIPCIONES)
        utiles = {int(c): d for c, d in descs.items() if not d.get("error")}

        if modo == "lexico":
            scores = {c: s for c, s in self._scores_lexico(consulta, utiles, True).items() if s > 0}
        elif modo == "texto":
            scores = self._scores_texto(out_dir, consulta, descs, utiles, progress)
        else:
            # El semantico aporta recall (tolera otra redaccion) y el lexico
            # precision (amarra cada color con su prenda). Normalizados, 50/50.
            sem = self._scores_texto(out_dir, consulta, descs, utiles, progress)
            lex = self._scores_lexico(consulta, utiles, False)
            ids = list(sem)
            if not lex or not any(lex.values()):
                scores = sem
            else:
                s_n = search.minmax(sem[c] for c in ids)
                l_n = search.minmax(lex.get(c, 0.0) for c in ids)
                scores = {c: 0.5 * a + 0.5 * b for c, a, b in zip(ids, s_n, l_n)}

        ranking = sorted(search.agregar_por_track(crops, scores).items(), key=lambda kv: -kv[1][0])[:top_k]
        tracks = {t["track_id"]: t for t in meta["tracks"]}
        resultados = []
        for pos, (tid, (score, cid)) in enumerate(ranking, 1):
            e = crops[cid]
            d = descs.get(str(cid), {})
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
