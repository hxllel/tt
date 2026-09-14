"""Busqueda: normalizacion de consultas tipo Alerta Amber, coincidencia lexica por
atributos y ranking por similitud coseno. Solo numpy: sin modelos aqui."""

import re
import unicodedata

import numpy as np

# Sinonimos -> termino canonico (espanol de Mexico). Incluye ingles porque el
# VLM a veces responde en ingles.
_SINONIMOS = {
    # prenda superior
    "playera": ["playera", "camiseta", "remera", "tshirt", "t-shirt", "polo", "top", "shirt"],
    "camisa": ["camisa", "blusa", "blouse"],
    "sudadera": ["sudadera", "hoodie", "buzo", "sweatshirt"],
    "chamarra": ["chamarra", "chaqueta", "campera", "jacket", "abrigo", "coat", "rompevientos"],
    "sueter": ["sueter", "jersey", "sweater", "chaleco", "vest"],
    "vestido": ["vestido", "dress"],
    "uniforme": ["uniforme", "uniform"],
    # prenda inferior
    "pantalon": ["pantalon", "pants", "trousers", "pantalones"],
    "mezclilla": ["mezclilla", "jeans", "jean", "vaquero", "vaqueros", "denim"],
    "pants": ["deportivo", "joggers", "sweatpants", "pans"],
    "short": ["short", "shorts", "bermuda", "bermudas"],
    "falda": ["falda", "skirt", "minifalda"],
    "leggins": ["leggins", "mallas", "leggings", "licras"],
    # calzado
    "tenis": ["tenis", "zapatillas", "sneakers"],
    "zapatos": ["zapatos", "zapato", "shoes", "calzado"],
    "botas": ["botas", "bota", "boots", "botines"],
    "sandalias": ["sandalias", "huaraches", "chanclas", "sandals", "flip-flops"],
    # accesorios
    "mochila": ["mochila", "backpack"],
    "gorra": ["gorra", "cachucha", "cap"],
    "sombrero": ["sombrero", "hat"],
    "lentes": ["lentes", "gafas", "anteojos", "glasses", "sunglasses"],
    "bolsa": ["bolsa", "bolso", "bag", "handbag"],
    "cubrebocas": ["cubrebocas", "mascarilla", "mask"],
    # colores
    "azul": ["azul", "blue", "marino", "navy", "celeste", "turquesa"],
    "rojo": ["rojo", "roja", "red", "guinda", "vino", "tinto"],
    "negro": ["negro", "negra", "black"],
    "blanco": ["blanco", "blanca", "white"],
    "gris": ["gris", "gray", "grey", "plateado", "plata"],
    "verde": ["verde", "green", "olivo"],
    "amarillo": ["amarillo", "amarilla", "yellow", "mostaza"],
    "cafe": ["cafe", "marron", "brown", "chocolate", "caqui", "khaki"],
    "rosa": ["rosa", "rosado", "rosada", "pink", "fiusha", "fucsia"],
    "morado": ["morado", "morada", "purple", "lila", "violeta"],
    "naranja": ["naranja", "anaranjado", "anaranjada", "orange"],
    "beige": ["beige", "crema", "hueso", "arena"],
}
CANON = {s: k for k, vs in _SINONIMOS.items() for s in vs}
COLORES = {"azul", "rojo", "negro", "blanco", "gris", "verde", "amarillo", "cafe", "rosa", "morado", "naranja", "beige"}
PRENDAS = set(_SINONIMOS) - COLORES
_SEPARADORES = re.compile(r"[,;.:\n]|\by\b|\bcon\b|\be\b|\band\b|\bwith\b")


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("t-shirt", "tshirt").replace("t shirt", "tshirt")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9,;.:\n ]", " ", texto)).strip()


def canon(token: str):
    if token in CANON:
        return CANON[token]
    for suf in ("es", "s"):  # plurales: azules, pantalones, tenis
        if token.endswith(suf) and token[: -len(suf)] in CANON:
            return CANON[token[: -len(suf)]]
    return None


def canon_tokens(texto: str) -> list:
    return [c for c in (canon(t) for t in normalizar(texto).replace(",", " ").split()) if c]


def frases_atributo(texto: str) -> list:
    """'Playera roja, pantalon de mezclilla azul y tenis blancos' ->
    [{prendas:{playera}, colores:{rojo}}, {prendas:{pantalon, mezclilla}, colores:{azul}}, ...]"""
    frases = []
    for chunk in _SEPARADORES.split(normalizar(texto)):
        toks = canon_tokens(chunk)
        prendas = {t for t in toks if t in PRENDAS}
        colores = {t for t in toks if t in COLORES}
        if prendas:
            frases.append({"prendas": prendas, "colores": colores, "texto": chunk.strip()})
        elif colores and frases:  # "... de color azul" se pega a la frase anterior
            frases[-1]["colores"] |= colores
    return frases


def score_lexico(frases_consulta: list, doc_texto: str) -> float:
    """Fraccion de atributos de la consulta presentes en el documento.
    Prenda con color correcto = 1, prenda sin color o con otro color = 0.5."""
    if not frases_consulta:
        return 0.0
    doc_frases = frases_atributo(doc_texto)
    doc_tokens = set(canon_tokens(doc_texto))
    total = 0.0
    for q in frases_consulta:
        mejor = 0.0
        for d in doc_frases:
            if q["prendas"] & d["prendas"]:
                mejor = max(mejor, 1.0 if (not q["colores"] or q["colores"] & d["colores"]) else 0.5)
        if mejor == 0.0 and q["prendas"] & doc_tokens:
            mejor = 0.5
        total += mejor
    return total / len(frases_consulta)


def textos_clip(consulta: str) -> list:
    """Plantilla completa + una por atributo; se promedian con penalizacion del minimo."""
    textos = [f"una foto de una persona que viste {consulta.strip()}"]
    textos += [f"una foto de una persona con {f['texto']}" for f in frases_atributo(consulta)]
    return textos


def score_clip(emb_imagenes: np.ndarray, emb_textos: np.ndarray) -> np.ndarray:
    """emb_imagenes (N, D) y emb_textos (1+F, D) normalizados -> score (N,)."""
    S = emb_imagenes @ emb_textos.T
    completo = S[:, 0]
    if S.shape[1] == 1:
        return completo
    atr = S[:, 1:]
    return 0.5 * completo + 0.5 * (0.5 * atr.mean(axis=1) + 0.5 * atr.min(axis=1))


def coseno(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    rng = x.max() - x.min() if x.size else 0
    return (x - x.min()) / rng if rng > 1e-9 else np.ones_like(x)


def agregar_por_track(crops: list, scores: dict) -> dict:
    """scores {crop_id: score} -> {track_id: (score_max, crop_id)}"""
    mejor = {}
    for cid, s in scores.items():
        tid = crops[cid]["track_id"]
        if tid not in mejor or s > mejor[tid][0]:
            mejor[tid] = (float(s), cid)
    return mejor
