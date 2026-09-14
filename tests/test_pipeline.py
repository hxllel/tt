import numpy as np
import pytest

from pipeline.detect_track import TrackState, expand_bbox, quality_score
from pipeline.describe import parse_respuesta, texto_de_busqueda
from pipeline.hardware import HardwareInfo, choose_profile, run_batched
from pipeline.metrics_yolo import iou_matrix, match_greedy
from pipeline.search import agregar_por_track, frases_atributo, normalizar, score_clip, score_lexico


# ------------------------------------------------------------------ busqueda
def test_normalizar_quita_acentos():
    assert normalizar("Pantalón Café, SUDADERA") == "pantalon cafe, sudadera"


def test_frases_atributo_alerta():
    f = frases_atributo("Vestía playera roja, pantalón de mezclilla azul y tenis blancos")
    assert [x["prendas"] for x in f] == [{"playera"}, {"pantalon", "mezclilla"}, {"tenis"}]
    assert [x["colores"] for x in f] == [{"rojo"}, {"azul"}, {"blanco"}]


def test_color_suelto_se_pega_a_frase_anterior():
    f = frases_atributo("sudadera de color gris; de color negro")
    assert f[0]["colores"] == {"gris", "negro"}


def test_lexico_sinonimos_y_color():
    q = frases_atributo("playera roja, jeans azules")
    assert score_lexico(q, "camiseta roja, pantalón de mezclilla azul") == 1.0
    assert score_lexico(q, "camiseta azul, pantalón de mezclilla rojo") == 0.5   # prendas bien, colores cruzados
    assert score_lexico(q, "vestido verde") == 0.0


def test_score_clip_ordena_por_coseno():
    t = np.eye(3, dtype=np.float32)            # texto completo + 2 atributos
    img = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    s = score_clip(img, t)
    assert s.argmax() == 0                     # coincide con todo
    assert s[1] > s[2]


def test_agregar_por_track_toma_maximo():
    crops = [{"track_id": 7}, {"track_id": 7}, {"track_id": 9}]
    assert agregar_por_track(crops, {0: 0.2, 1: 0.8, 2: 0.5}) == {7: (0.8, 1), 9: (0.5, 2)}


# ------------------------------------------------------------------ deteccion
def test_expand_bbox_clamp():
    assert expand_bbox(0, 0, 100, 200, 105, 300, 0.10) == (0, 0, 105, 220)


def test_quality_prefiere_grande_y_sin_borde():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (250, 100, 3), dtype=np.uint8)
    q_centro = quality_score(crop, (100, 100, 200, 350), 1920, 1080, 0.9)
    q_borde = quality_score(crop, (0, 100, 100, 350), 1920, 1080, 0.9)
    q_chico = quality_score(crop[:60], (100, 100, 124, 160), 1920, 1080, 0.9)
    assert q_centro > q_borde and q_centro > q_chico


def test_trackstate_topk():
    ts = TrackState(1, k=2)
    for i, q in enumerate([0.1, 0.9, 0.5, 0.3]):
        ts.add(q, i, float(i), i, {"i": i}, np.zeros((2, 2, 3)))
    assert [x[0] for x in ts.best()] == [0.9, 0.5]
    assert ts.n_det == 4 and ts.first_frame == 0 and ts.last_frame == 3


# ------------------------------------------------------------------ hardware
def _gpu(free):
    return HardwareInfo(device="cuda", vram_free_gb=free, vram_total_gb=free)


def test_perfil_por_vram():
    assert choose_profile(_gpu(3)).vlm == "0.5b"
    p = choose_profile(_gpu(6))
    assert (p.vlm, p.yolo, p.vlm_batch) == ("1.5b", "yolo26s", 8)
    assert choose_profile(_gpu(24)).vlm == "7b"
    assert choose_profile(HardwareInfo(device="cpu")).half is False


def test_perfil_respeta_checkpoints_disponibles():
    assert choose_profile(_gpu(24), ["0.5b", "1.5b"]).vlm == "1.5b"
    assert choose_profile(_gpu(3), ["1.5b"]).vlm == "1.5b"


def test_run_batched_reduce_lote_con_oom():
    tamanos = []

    def fn(chunk):
        tamanos.append(len(chunk))
        if len(chunk) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return [x * 2 for x in chunk]

    res, lote = run_batched(fn, list(range(7)), 8)
    assert res == [x * 2 for x in range(7)] and lote == 2


def test_run_batched_propaga_otros_errores():
    with pytest.raises(RuntimeError, match="otro"):
        run_batched(lambda c: (_ for _ in ()).throw(RuntimeError("otro")), [1, 2], 2)


# ------------------------------------------------------------------ describe / metricas
def test_parse_respuesta_json_con_ruido():
    txt = 'Claro: {"superior_tipo": "playera", "superior_color": "roja", "accesorios": ["mochila"]} fin'
    d = parse_respuesta(txt)
    assert d["superior_color"] == "roja"
    assert "playera roja" in texto_de_busqueda({"atributos": d, "descripcion": "x"})
    assert parse_respuesta("sin json") is None


def test_match_greedy():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    pred = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    ok, tp, fp = match_greedy(gt, pred, np.array([0.9, 0.8]))
    assert ok.tolist() == [True, False] and tp == 1 and fp == 1
    assert iou_matrix(gt, pred)[0, 0] == pytest.approx(1.0)


def test_limpiar_salida_primera_linea():
    from pipeline.describe import limpiar_salida
    txt = "The person is wearing a white shirt.\nWhat is the person wearing?\nAnswer: x"
    assert limpiar_salida(txt) == "The person is wearing a white shirt."
