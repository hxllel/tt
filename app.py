"""Interfaz grafica del prototipo TT 2027-A048 (Gradio, solo local).

    conda activate tt-a048
    python app.py                      # autodetecta GPU; abre http://127.0.0.1:7860
    python app.py --dispositivo cpu    # fuerza CPU

En un servidor remoto con GPU usa un tunel SSH (ssh -L 7860:localhost:7860 usuario@host);
nunca share=True: los videos contienen datos personales (RNF-14/15).
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import shutil
from pathlib import Path

# pipeline.config fija HF_HOME=models/hf; debe importarse antes que gradio/huggingface_hub
from pipeline.config import load_config, ruta, videos_procesados

import gradio as gr
import pandas as pd

from pipeline import metrics_yolo
from pipeline.hardware import resumen
from pipeline.runner import MODOS, Motor
from pipeline.video_io import fmt_tiempo

AVISO = ("⚠️ Cada coincidencia es una **sugerencia sujeta a verificación humana**, no una identificación "
         "positiva (RNF-12). El sistema solo compara vestimenta; no usa rostros ni datos biométricos.")
MODOS_UI = {
    "CLIP imagen↔texto (rápido, recomendado con miles de personas)": "clip",
    "Híbrido: CLIP + verificación con FastVLM (top-K)": "hibrido",
    "Texto↔texto: descripción FastVLM vs alerta (e5)": "texto",
    "Léxico por atributos (línea base TT1)": "lexico",
}
MOTOR: Motor = None


def _videos():
    return gr.update(choices=videos_procesados(MOTOR.cfg))


# ------------------------------------------------------------------ Sistema
def aplicar_ajustes(dispositivo, yolo, imgsz, intervalo, sr, vlm, resolucion, max_tokens, formato):
    MOTOR.reconfigurar({
        "dispositivo": dispositivo,
        "deteccion": {"modelo": yolo, "imgsz": imgsz if imgsz == "auto" else int(imgsz), "intervalo_frames": int(intervalo)},
        "superres": {"activar": sr},
        "vlm": {"modelo": vlm, "resolucion": int(resolucion), "max_new_tokens": int(max_tokens), "formato": formato},
    })
    return resumen(MOTOR.info, MOTOR.perfil)


def exportar():
    try:
        return f"Exportado: `{MOTOR.exportar_detector()}`. Se usará automáticamente."
    except Exception as e:
        raise gr.Error(f"No se pudo exportar: {e}")


# ------------------------------------------------------------------ Procesar
def procesar(archivos, rutas_texto, procedencia, consentimiento, notas, describir, progress=gr.Progress()):
    videos = []
    destino = ruta(MOTOR.cfg["videos_dir"])
    destino.mkdir(parents=True, exist_ok=True)
    for f in archivos or []:
        p = Path(f)
        final = destino / p.name
        if not final.exists():
            shutil.copy(p, final)
        videos.append(final)
    for linea in (rutas_texto or "").splitlines():
        if linea.strip():
            p = Path(linea.strip().strip('"'))
            if not p.exists():
                raise gr.Error(f"No existe: {p}")
            videos.append(p)
    if not videos:
        raise gr.Error("Sube un video o escribe su ruta.")
    if procedencia == "Voluntario propio" and not consentimiento:
        raise gr.Error("Registra el consentimiento informado por escrito antes de procesar (RF-17).")

    filas = []
    for v in videos:
        st = MOTOR.procesar_video(v, {"origen": procedencia, "consentimiento": bool(consentimiento), "notas": notas},
                                  describir, progress)
        d = st.get("deteccion", {})
        filas.append({
            "video": v.stem, "personas": d.get("tracks_guardados"), "recortes": d.get("recortes"),
            "máx. en frame": d.get("max_personas_en_frame"),
            "detección (s)": d.get("tiempo_deteccion_s"), "fps proc.": d.get("fps_procesados"),
            "SR (s)": st.get("superres", {}).get("tiempo_s"), "CLIP (s)": st.get("clip", {}).get("tiempo_s"),
            "FastVLM (s)": st.get("descripcion", {}).get("tiempo_s"),
        })
    return pd.DataFrame(filas), _videos(), _videos()


def describir_todo(video, progress=gr.Progress()):
    if not video:
        raise gr.Error("Elige un video procesado.")
    descs = MOTOR.describir(video, None, progress)
    st = MOTOR.stats(video).get("descripcion", {})
    return f"{len(descs)} descripciones. Última corrida: {st}"


# ------------------------------------------------------------------ Buscar
def buscar(video, consulta, modo_ui, top_k, progress=gr.Progress()):
    if not video:
        raise gr.Error("Elige un video procesado.")
    try:
        res = MOTOR.buscar(video, consulta, MODOS_UI[modo_ui], int(top_k), progress)
    except (ValueError, FileNotFoundError) as e:
        raise gr.Error(str(e))
    if not res:
        gr.Warning("Sin coincidencias.")
    galeria = [(r["imagen"], f"#{r['posicion']} · {fmt_tiempo(r['timestamp_sec'])} · {r['score']:.3f}") for r in res]
    tabla = pd.DataFrame([{
        "#": r["posicion"], "track": r["track_id"], "score": r["score"],
        "aparición": fmt_tiempo(r["timestamp_sec"]),
        "en cámara": f"{fmt_tiempo(r['t_inicio'] or 0)}–{fmt_tiempo(r['t_fin'] or 0)}",
        "descripción": r["descripcion"],
    } for r in res])
    return galeria, tabla, res, None, [], ""


def seleccionar(res, video, evt: gr.SelectData):
    if not res or evt.index is None:
        return None, [], ""
    r = res[evt.index]
    detalle = (f"**Track {r['track_id']}** · aparición {fmt_tiempo(r['timestamp_sec'])} "
               f"(frame {r['frame_idx']}) · en cámara {fmt_tiempo(r['t_inicio'] or 0)}–{fmt_tiempo(r['t_fin'] or 0)} "
               f"· score {r['score']}  \n{r['descripcion'] or '_sin descripción FastVLM_'}")
    return MOTOR.escena(video, r["crop_id"]), MOTOR.tira(video, r["crop_id"]), detalle


# ------------------------------------------------------------------ Métricas
def metricas_stats(video):
    if not video:
        raise gr.Error("Elige un video procesado.")
    res, png = metrics_yolo.estadisticas_sin_gt(MOTOR.cfg, video)
    return res, str(png)


def metricas_dataset(data_yaml, conf):
    if not data_yaml or not Path(data_yaml).exists():
        raise gr.Error("Indica la ruta a un dataset.yaml en formato YOLO (clase 0 = persona).")
    return metrics_yolo.evaluar_dataset(MOTOR.cfg, MOTOR.device, MOTOR.yolo, data_yaml, float(conf))


def metricas_mot(video_path, gt_path, intervalo):
    for p in (video_path, gt_path):
        if not p or not Path(p).exists():
            raise gr.Error(f"No existe: {p}")
    return metrics_yolo.evaluar_mot(MOTOR.cfg, MOTOR.device, MOTOR.yolo, video_path, gt_path, int(intervalo))


# ------------------------------------------------------------------ UI
def construir() -> gr.Blocks:
    cfg = MOTOR.cfg
    with gr.Blocks(title="TT 2027-A048 · Búsqueda de personas") as demo:
        gr.Markdown("# Búsqueda de personas en grabaciones por descripción de vestimenta\n"
                    "Prototipo offline TT 2027-A048 · YOLO26 + ByteTrack → superresolución → CLIP / FastVLM")
        gr.Markdown(AVISO)

        with gr.Tab("1. Procesar video"):
            with gr.Row():
                with gr.Column():
                    archivos = gr.File(label="Videos", file_count="multiple", type="filepath",
                                       file_types=["video"])
                    rutas = gr.Textbox(label="…o rutas en disco (una por línea, evita subir archivos grandes)", lines=2)
                with gr.Column():
                    procedencia = gr.Dropdown(["Voluntario propio", "Dataset público", "Otro"],
                                              value="Voluntario propio", label="Procedencia (RF-17)")
                    consentimiento = gr.Checkbox(label="Consentimiento informado por escrito registrado")
                    notas = gr.Textbox(label="Notas", lines=1)
                    describir = gr.Checkbox(value=MOTOR.device == "cuda",
                                            label="Describir a todas las personas con FastVLM (lento en CPU)")
            btn_proc = gr.Button("Procesar", variant="primary")
            tabla_proc = gr.Dataframe(label="Resultado", interactive=False)

        with gr.Tab("2. Buscar"):
            with gr.Row():
                video = gr.Dropdown(videos_procesados(cfg), label="Video procesado", scale=2)
                refrescar = gr.Button("↻", scale=0)
            consulta = gr.Textbox(label="Descripción de la alerta (tal como aparece en la Alerta Amber / cédula)",
                                  placeholder="Vestía playera roja, pantalón de mezclilla azul, tenis blancos y mochila negra",
                                  lines=2)
            with gr.Row():
                modo = gr.Dropdown(list(MODOS_UI), value=list(MODOS_UI)[0], label="Modo de búsqueda", scale=3)
                top_k = gr.Slider(1, 100, value=cfg["busqueda"]["top_k"], step=1, label="Top-K", scale=1)
            with gr.Row():
                btn_buscar = gr.Button("Buscar", variant="primary")
                btn_desc = gr.Button("Generar descripciones FastVLM de todo el video")
            estado_desc = gr.Markdown()
            galeria = gr.Gallery(label="Coincidencias (clic para ver la escena)", columns=6, height=360,
                                 object_fit="contain")
            tabla = gr.Dataframe(interactive=False, wrap=True)
            detalle = gr.Markdown()
            with gr.Row():
                escena = gr.Image(label="Escena", type="numpy", scale=3)
                tira = gr.Gallery(label="Contexto ±3 s", columns=3, scale=2, height=360)
            resultados = gr.State([])

        with gr.Tab("3. Métricas YOLO"):
            gr.Markdown("**(a)** Dataset anotado (formato YOLO, clase 0 = persona; exporta desde CVAT o Label Studio) → "
                        "P, R, mAP y % de personas detectadas.  \n**(b)** Ground truth MOT → MOTA, IDF1, cambios de ID.  \n"
                        "**(c)** Sin anotaciones → solo estadísticas (no mide exactitud).")
            with gr.Tab("(c) Estadísticas del video"):
                video_m = gr.Dropdown(videos_procesados(cfg), label="Video procesado")
                btn_m = gr.Button("Calcular")
                json_m = gr.JSON()
                img_m = gr.Image(type="filepath")
            with gr.Tab("(a) Dataset anotado"):
                data_yaml = gr.Textbox(label="Ruta a dataset.yaml")
                conf_op = gr.Slider(0.05, 0.95, value=cfg["deteccion"]["conf_recorte"], label="Confianza de operación")
                btn_a = gr.Button("Evaluar")
                json_a = gr.JSON()
            with gr.Tab("(b) Tracking MOT"):
                mot_video = gr.Textbox(label="Ruta al video")
                mot_gt = gr.Textbox(label="Ruta a gt.txt (frame,id,x,y,w,h,...)")
                mot_int = gr.Number(value=1, precision=0, label="Intervalo de frames")
                btn_b = gr.Button("Evaluar")
                json_b = gr.JSON()

        with gr.Tab("4. Sistema"):
            info_md = gr.Markdown(resumen(MOTOR.info, MOTOR.perfil))
            with gr.Row():
                dispositivo = gr.Radio(["auto", "cuda", "cpu"], value=cfg["dispositivo"], label="Dispositivo")
                yolo = gr.Dropdown(["auto", "yolo26n", "yolo26s", "yolo26m", "yolo26l"],
                                   value=cfg["deteccion"]["modelo"], label="Detector")
                imgsz = gr.Dropdown(["auto", 640, 960, 1280], value=cfg["deteccion"]["imgsz"], label="imgsz YOLO")
                intervalo = gr.Number(value=cfg["deteccion"]["intervalo_frames"], precision=0, label="Intervalo de frames")
            with gr.Row():
                sr = gr.Radio(["auto", "true", "false"], value=str(cfg["superres"]["activar"]).lower(),
                              label="Superresolución")
                vlm = gr.Dropdown(["auto", "0.5b", "1.5b", "7b"], value=cfg["vlm"]["modelo"], label="FastVLM")
                resolucion = gr.Dropdown([1024, 768, 512], value=cfg["vlm"]["resolucion"], label="Resolución codificador")
                max_tokens = gr.Number(value=cfg["vlm"]["max_new_tokens"], precision=0, label="max_new_tokens")
                formato = gr.Radio(["en", "json", "libre"], value=cfg["vlm"]["formato"], label="Prompt FastVLM")
            with gr.Row():
                btn_aplicar = gr.Button("Aplicar")
                btn_export = gr.Button("Exportar YOLO optimizado (TensorRT en GPU / OpenVINO en CPU)")
            export_md = gr.Markdown()

        btn_proc.click(procesar, [archivos, rutas, procedencia, consentimiento, notas, describir],
                       [tabla_proc, video, video_m])
        refrescar.click(lambda: (_videos(), _videos()), None, [video, video_m])
        btn_buscar.click(buscar, [video, consulta, modo, top_k], [galeria, tabla, resultados, escena, tira, detalle])
        consulta.submit(buscar, [video, consulta, modo, top_k], [galeria, tabla, resultados, escena, tira, detalle])
        btn_desc.click(describir_todo, [video], [estado_desc])
        galeria.select(seleccionar, [resultados, video], [escena, tira, detalle])
        btn_m.click(metricas_stats, [video_m], [json_m, img_m])
        btn_a.click(metricas_dataset, [data_yaml, conf_op], [json_a])
        btn_b.click(metricas_mot, [mot_video, mot_gt, mot_int], [json_b])
        btn_aplicar.click(aplicar_ajustes, [dispositivo, yolo, imgsz, intervalo, sr, vlm, resolucion, max_tokens, formato],
                          [info_md])
        btn_export.click(exportar, None, [export_md])
    return demo


def main():
    global MOTOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-abrir", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, {"dispositivo": args.dispositivo} if args.dispositivo else None)
    MOTOR = Motor(cfg)
    print(resumen(MOTOR.info, MOTOR.perfil).replace("**", "").replace("  \n", "\n"))
    permitidos = [str(ruta(cfg["salidas_dir"])), str(ruta(cfg["videos_dir"]))]
    for p in permitidos:
        Path(p).mkdir(parents=True, exist_ok=True)
    construir().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, inbrowser=not args.no_abrir,
        share=False, allowed_paths=permitidos)


if __name__ == "__main__":
    main()
