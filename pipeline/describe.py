"""Etapa 2: descripcion de recortes con FastVLM cargado UNA vez en el proceso.

Frente a search_fastvlm.py original (un subprocess por recorte que recargaba el
modelo) aqui se procesa por lotes, en greedy, con salida corta y medicion de
TTFT por lote."""

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from pipeline.config import ruta
from pipeline.detect_track import _actualizar_stats, cargar_metadata
from pipeline.hardware import is_oom, run_batched

# FastVLM se entreno con instrucciones en ingles: en espanol o pidiendo JSON el
# modelo 0.5B ignora la instruccion y alucina. La busqueda ya mapea terminos en
# ingles (search.py) y e5 es multilingue.
PROMPTS = {
    "en": ("Describe the clothing of this person in one short sentence: upper garment and its color, "
           "lower garment and its color, shoes, and any bag, backpack or hat."),
    "json": ('Describe la ropa de la persona de la imagen. Responde solo con un JSON con las claves '
             '"superior_tipo", "superior_color", "inferior_tipo", "inferior_color", "calzado", '
             '"accesorios" (lista) y "descripcion" (una frase en espanol).'),
    "libre": ("Describe a la persona en la imagen en espanol: prenda superior y su color, prenda "
              "inferior y su color, calzado y accesorios visibles."),
}


class _FirstTokenTimer(LogitsProcessor):
    """Se llama por primera vez justo despues del prefill: marca el TTFT."""

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.t = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            if self.cuda:
                torch.cuda.synchronize()
            self.t = time.perf_counter()
        return scores


def encontrar_checkpoint(path) -> Path:
    """Acepta la carpeta del checkpoint o una carpeta que la contiene (zip anidado)."""
    path = ruta(path)
    if (path / "config.json").exists():
        return path
    for sub in sorted(path.glob("*/config.json")) if path.exists() else []:
        if "__MACOSX" not in str(sub):
            return sub.parent
    raise FileNotFoundError(f"No se encontro un checkpoint de FastVLM en {path}")


def checkpoints_disponibles(cfg: dict) -> list:
    disp = []
    for k, p in cfg["vlm"]["checkpoints"].items():
        try:
            encontrar_checkpoint(p)
            disp.append(k)
        except FileNotFoundError:
            pass
    return disp


def parse_respuesta(texto: str):
    ini, fin = texto.find("{"), texto.rfind("}")
    if ini == -1 or fin <= ini:
        return None
    frag = texto[ini:fin + 1]
    for cand in (frag, frag.replace("'", '"')):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def limpiar_salida(txt: str) -> str:
    """El 0.5B suele continuar con 'What is the person wearing?', 'Answer:' o '<start of ...'
    tras la primera frase: se conserva solo la primera linea util."""
    limpio = txt.strip().split("\n")[0].split("Answer:")[0].split("<start")[0].strip()
    return limpio or txt.strip()


def texto_de_busqueda(rec: dict) -> str:
    """Texto plano usado por la busqueda lexica y por los embeddings de texto."""
    a = rec.get("atributos") or {}
    partes = [f"{a.get('superior_tipo', '')} {a.get('superior_color', '')}",
              f"{a.get('inferior_tipo', '')} {a.get('inferior_color', '')}",
              str(a.get("calzado", ""))]
    acc = a.get("accesorios") or []
    partes += acc if isinstance(acc, list) else [str(acc)]
    partes.append(rec.get("descripcion") or rec.get("raw") or "")
    return ", ".join(p.strip() for p in partes if p and p.strip())


class Describer:
    def __init__(self, cfg: dict, device: str, vlm: str, batch: int):
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        vcfg = cfg["vlm"]
        self.cfg = vcfg
        self.cuda = device == "cuda"
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.dtype = torch.float16 if self.cuda else torch.float32
        self.batch = batch
        self.vlm = vlm
        ckpt = encontrar_checkpoint(vcfg["checkpoints"][vlm])

        # predict.py renombra generation_config.json; aqui no hace falta porque
        # todos los parametros de generacion se pasan explicitamente.
        t0 = time.perf_counter()
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(ckpt), None, get_model_name_from_path(str(ckpt)),
            device="cuda" if self.cuda else "cpu",
            device_map="cuda:0" if self.cuda else "auto",
        )
        self.model.eval()
        if not self.cuda:
            torch.set_flush_denormal(True)  # pesos fp16->fp32 con denormales frenan la CPU
            self.model.float()
            if vcfg.get("cpu_int8", False):
                torch.ao.quantization.quantize_dynamic(
                    self.model.get_model().layers, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        self.load_s = time.perf_counter() - t0

        res = int(vcfg["resolucion"])
        self.image_processor.crop_size = {"height": res, "width": res}
        self.image_processor.size = {"shortest_edge": res}

        self.tokenizer.padding_side = "left"
        self.model.config.tokenizer_padding_side = "left"
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        qs = PROMPTS[vcfg["formato"]]
        img_tok = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                   if self.model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN)
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], img_tok + "\n" + qs)
        conv.append_message(conv.roles[1], None)
        self.input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                               return_tensors="pt").unsqueeze(0)

    @torch.inference_mode()
    def generate(self, images: list, max_new_tokens: int = None) -> list:
        from llava.mm_utils import process_images
        tensor = process_images(images, self.image_processor, self.model.config)
        if isinstance(tensor, list):
            tensor = torch.stack(tensor)
        tensor = tensor.to(self.device, dtype=self.dtype)
        ids = self.input_ids.repeat(len(images), 1).to(self.device)
        timer = _FirstTokenTimer(self.cuda)
        if self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(
            ids, images=tensor, image_sizes=[im.size for im in images],
            do_sample=False, num_beams=1, temperature=None, top_p=None,
            max_new_tokens=max_new_tokens or self.cfg["max_new_tokens"], use_cache=True,
            logits_processor=LogitsProcessorList([timer]),
        )
        if self.cuda:
            torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        ttft = (timer.t or time.perf_counter()) - t0
        pad = self.tokenizer.pad_token_id
        textos = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        res = []
        for row, txt in zip(out, textos):
            txt = txt.strip() if self.cfg["formato"] == "json" else limpiar_salida(txt)
            attrs = parse_respuesta(txt) if self.cfg["formato"] == "json" else None
            rec = {"raw": txt, "atributos": attrs,
                   "descripcion": (attrs or {}).get("descripcion") or txt,
                   "tokens": int((row != pad).sum()), "ttft_s": round(ttft, 4),
                   "gen_s": round(gen_s, 4), "lote": len(images), "error": None}
            rec["texto_busqueda"] = texto_de_busqueda(rec)
            res.append(rec)
        return res

    def describe_paths(self, paths: list, progress=None):
        """Describe una lista de rutas. Devuelve (registros, lote_final)."""
        def lote(chunk):
            imgs, errores = [], {}
            for i, p in enumerate(chunk):
                try:
                    imgs.append((i, Image.open(p).convert("RGB")))
                except Exception as e:
                    errores[i] = f"no se pudo abrir: {e}"
            salida = [None] * len(chunk)
            for i, msg in errores.items():
                salida[i] = {"error": msg}
            if imgs:
                try:
                    recs = self.generate([im for _, im in imgs])
                    for (i, _), r in zip(imgs, recs):
                        salida[i] = r
                except Exception as e:
                    if is_oom(e):
                        raise
                    for i, im in imgs:  # aisla el recorte problematico (RF-16)
                        try:
                            salida[i] = self.generate([im])[0]
                        except Exception as e2:
                            if is_oom(e2):
                                raise
                            salida[i] = {"error": str(e2)}
            return salida

        cb = (lambda i, n: progress(i / n, f"FastVLM {self.vlm}: {i}/{n} recortes")) if progress is not None else None
        recs, self.batch = run_batched(lote, list(paths), self.batch, cb)
        return recs, self.batch


def ruta_recorte(out_dir: Path, entry: dict) -> Path:
    return out_dir / (entry.get("crop_sr_path") or entry["crop_path"])


def cargar_descripciones(out_dir) -> dict:
    p = Path(out_dir) / "descripciones.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def describir_video(out_dir, describer: Describer, crop_ids=None, progress=None) -> dict:
    """Describe los recortes indicados (por defecto el mejor de cada track) y los
    agrega a descripciones.json sin repetir los ya descritos."""
    out_dir = Path(out_dir)
    meta = cargar_metadata(out_dir)
    descs = cargar_descripciones(out_dir)
    crops = meta["crops"]
    if crop_ids is None:
        crop_ids = [e["crop_id"] for e in crops if e["rank"] == 0]
    faltan = [c for c in crop_ids if str(c) not in descs]
    if not faltan:
        return descs
    t0 = time.perf_counter()
    recs, lote = describer.describe_paths([ruta_recorte(out_dir, crops[c]) for c in faltan], progress)
    for c, r in zip(faltan, recs):
        descs[str(c)] = {"crop_id": c, "track_id": crops[c]["track_id"], **r}
    (out_dir / "descripciones.json").write_text(json.dumps(descs, ensure_ascii=False, indent=1), encoding="utf-8")

    dur = time.perf_counter() - t0
    ok = [r for r in recs if not r.get("error")]
    tokens = sum(r["tokens"] for r in ok)
    _actualizar_stats(out_dir, {"descripcion": {
        "modelo": describer.vlm, "dispositivo": "cuda" if describer.cuda else "cpu",
        "resolucion": describer.cfg["resolucion"], "recortes": len(faltan), "errores": len(recs) - len(ok),
        "lote_final": lote, "tiempo_s": round(dur, 2), "recortes_por_s": round(len(faltan) / dur, 3) if dur else 0,
        "tokens_por_s": round(tokens / dur, 2) if dur else 0,
        "ttft_medio_s": round(sum(r["ttft_s"] for r in ok) / len(ok), 4) if ok else None,
        "carga_modelo_s": round(describer.load_s, 2),
    }})
    return descs
