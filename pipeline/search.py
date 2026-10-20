"""Busqueda sobre las descripciones que genero FastVLM: normalizacion de consultas
tipo Alerta Amber, coincidencia lexica por atributos y utilidades de ranking.
Sin modelos ni dependencias pesadas. El sistema no usa CLIP: la unica lectura de
la imagen la hace FastVLM, y aqui solo se trabaja con su texto."""

import re
import unicodedata

# Sinonimos -> termino canonico (espanol de Mexico). Incluye ingles porque el
# prompt de FastVLM es en ingles y sus descripciones salen en ese idioma.
_SINONIMOS = {
    # prenda superior
    "playera": ["playera", "camiseta", "remera", "tshirt", "t-shirt", "polo", "top", "shirt"],
    "camisa": ["camisa", "blusa", "blouse"],
    "sudadera": ["sudadera", "hoodie", "buzo", "sweatshirt"],
    "chamarra": ["chamarra", "chaqueta", "campera", "jacket", "abrigo", "coat", "rompevientos",
                 "blazer", "saco", "gabardina"],
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
# Termino canonico -> ingles, para traducir la consulta antes de vectorizarla:
# FastVLM describe en ingles y el emparejamiento mejora bastante.
EN_CANON = {
    "playera": "t-shirt", "camisa": "shirt", "sudadera": "hoodie", "chamarra": "jacket",
    "sueter": "sweater", "vestido": "dress", "uniforme": "uniform",
    "pantalon": "pants", "mezclilla": "jeans", "pants": "sweatpants", "short": "shorts",
    "falda": "skirt", "leggins": "leggings",
    "tenis": "sneakers", "zapatos": "shoes", "botas": "boots", "sandalias": "sandals",
    "mochila": "backpack", "gorra": "cap", "sombrero": "hat", "lentes": "glasses",
    "bolsa": "bag", "cubrebocas": "face mask",
    "azul": "blue", "rojo": "red", "negro": "black", "blanco": "white", "gris": "gray",
    "verde": "green", "amarillo": "yellow", "cafe": "brown", "rosa": "pink",
    "morado": "purple", "naranja": "orange", "beige": "beige",
}
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


def consulta_en_ingles(texto: str) -> str:
    """"Chamarra negra y pantalon blanco" -> "black jacket, white pants".

    Devuelve "" si no se reconocio ninguna prenda; en ese caso conviene
    vectorizar la consulta original."""
    partes = []
    for f in frases_atributo(texto):
        prendas = " ".join(EN_CANON[p] for p in sorted(f["prendas"]) if p in EN_CANON)
        colores = " ".join(EN_CANON[c] for c in sorted(f["colores"]) if c in EN_CANON)
        if prendas:
            partes.append(f"{colores} {prendas}".strip())
    return ", ".join(partes)


def minmax(valores) -> list:
    """Escala a [0, 1] para poder promediar scores de escalas distintas."""
    valores = list(valores)
    if not valores:
        return []
    lo, hi = min(valores), max(valores)
    return [1.0] * len(valores) if hi - lo < 1e-9 else [(v - lo) / (hi - lo) for v in valores]


def agregar_por_track(crops: list, scores: dict) -> dict:
    """scores {crop_id: score} -> {track_id: (score_max, crop_id)}"""
    mejor = {}
    for cid, s in scores.items():
        tid = crops[cid]["track_id"]
        if tid not in mejor or s > mejor[tid][0]:
            mejor[tid] = (float(s), cid)
    return mejor
