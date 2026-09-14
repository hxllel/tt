"""
Etapa 2: Super-resolucion de recortes con HAT
--------------------------------------------------------------
Toma la salida de extract_bounding_boxes.py (una carpeta por
video, y dentro una carpeta por cada persona unica con todos
sus fotogramas recortados) y aplica super-resolucion HAT
(Hybrid Attention Transformer) a cada imagen, replicando
EXACTAMENTE la misma estructura de carpetas en OUTPUT_ROOT.

Requisitos previos:
    1. Clonar el repo oficial de HAT:
        git clone https://github.com/XPixelGroup/HAT.git
    2. Instalar dependencias dentro de ese repo:
        cd HAT
        pip install -r requirements.txt
        python setup.py develop
       (esto instala basicsr y registra las arquitecturas de HAT)
    3. Descargar un checkpoint pre-entrenado de HAT, por ejemplo
       HAT_SRx4_ImageNet-pretrain.pth, desde:
        https://github.com/XPixelGroup/HAT (seccion de modelos)
       y colocarlo en MODEL_PATH.
    4. Ajustar HAT_REPO_PATH abajo para que apunte a la carpeta
       donde clonaste el repo de HAT.

IMPORTANTE sobre NETWORK_CONFIG:
    Los parametros de NETWORK_CONFIG deben coincidir con el
    checkpoint que uses. Los valores por defecto de este script
    corresponden al modelo base "HAT_SRx4" (no HAT-S, no HAT-L,
    no las variantes GAN Real_HAT_GAN_*). Si usas otro checkpoint,
    copia el bloque `network_g` del .yml correspondiente en
    HAT/options/test/ y reemplaza NETWORK_CONFIG con esos valores.

Uso:
    python3 super_resolve_hat.py
(edita INPUT_ROOT, OUTPUT_ROOT, HAT_REPO_PATH y MODEL_PATH abajo)
"""

import os
import sys
import shutil
import cv2
import numpy as np
import torch

# ---------------- CONFIGURACION ----------------
INPUT_ROOT = "/home/hillel/Escritorio/tt/bounding_boxes"
OUTPUT_ROOT = "/home/hillel/Escritorio/tt/bounding_boxes_sr_hat"

HAT_REPO_PATH = "/home/hillel/Escritorio/tt/HAT"     # carpeta donde clonaste XPixelGroup/HAT
MODEL_PATH = "/home/hillel/Escritorio/tt/models/HAT_SRx4_ImageNet-pretrain.pth"

SCALE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si las personas recortadas son grandes y no te alcanza la VRAM,
# activa el modo tile (procesa la imagen en parches). None = sin tiling.
TILE_SIZE = None      # ej. 256
TILE_PAD = 16

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Arquitectura del modelo base HAT_SRx4 (ajustar si usas otro checkpoint,
# ver nota en el docstring de arriba).
NETWORK_CONFIG = dict(
    upscale=SCALE,
    in_chans=3,
    img_size=64,
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.0,
    depths=(6, 6, 6, 6, 6, 6),
    embed_dim=180,
    num_heads=(6, 6, 6, 6, 6, 6),
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)
# -------------------------------------------------


def load_hat_model():
    """Carga la arquitectura HAT y los pesos pre-entrenados."""
    sys.path.insert(0, HAT_REPO_PATH)
    try:
        from hat.archs.hat_arch import HAT
    except ImportError as e:
        raise ImportError(
            "No se pudo importar HAT desde HAT_REPO_PATH. Verifica que "
            "clonaste https://github.com/XPixelGroup/HAT y que corriste "
            "'python setup.py develop' dentro de ese repo, y que "
            "HAT_REPO_PATH apunta a esa carpeta."
        ) from e

    model = HAT(**NETWORK_CONFIG)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(DEVICE)
    return model


def _pad_to_multiple(img_tensor, multiple):
    """Padding por reflexion para que H y W sean multiplos de `multiple`
    (requerido por la ventana de atencion de HAT)."""
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    img_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return img_padded, h, w


@torch.no_grad()
def _infer_full(model, img_tensor, window_size):
    padded, orig_h, orig_w = _pad_to_multiple(img_tensor, window_size)
    output = model(padded)
    output = output[:, :, : orig_h * SCALE, : orig_w * SCALE]
    return output


@torch.no_grad()
def _infer_tiled(model, img_tensor, window_size, tile_size, tile_pad):
    """Inferencia por tiles para imagenes grandes con poca VRAM."""
    _, c, h, w = img_tensor.shape
    output = torch.zeros((1, c, h * SCALE, w * SCALE), device=img_tensor.device)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0, y1 = y, min(y + tile_size, h)
            x0, x1 = x, min(x + tile_size, w)

            y0_pad = max(0, y0 - tile_pad)
            x0_pad = max(0, x0 - tile_pad)
            y1_pad = min(h, y1 + tile_pad)
            x1_pad = min(w, x1 + tile_pad)

            tile = img_tensor[:, :, y0_pad:y1_pad, x0_pad:x1_pad]
            tile_out = _infer_full(model, tile, window_size)

            out_y0 = (y0 - y0_pad) * SCALE
            out_x0 = (x0 - x0_pad) * SCALE
            out_y1 = out_y0 + (y1 - y0) * SCALE
            out_x1 = out_x0 + (x1 - x0) * SCALE

            output[:, :, y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = (
                tile_out[:, :, out_y0:out_y1, out_x0:out_x1]
            )

    return output


def super_resolve_image(model, img_bgr, window_size=16, tile_size=TILE_SIZE, tile_pad=TILE_PAD):
    """Aplica HAT a una imagen BGR (numpy, uint8) y devuelve BGR uint8."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    if tile_size is not None:
        output = _infer_tiled(model, img_tensor, window_size, tile_size, tile_pad)
    else:
        output = _infer_full(model, img_tensor, window_size)

    output = output.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    output = (output * 255.0).round().astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output_bgr


def process_tree(model, input_root=INPUT_ROOT, output_root=OUTPUT_ROOT):
    """Recorre INPUT_ROOT (video/persona_X/*.jpg) y replica la misma
    estructura en OUTPUT_ROOT, aplicando HAT a cada imagen."""
    n_procesadas = 0
    n_saltadas = 0

    for dirpath, _dirnames, filenames in os.walk(input_root):
        rel_dir = os.path.relpath(dirpath, input_root)
        out_dir = os.path.join(output_root, rel_dir) if rel_dir != "." else output_root

        for filename in filenames:
            in_path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".json":
                # Copia metadata.json ajustando las rutas de crop_path
                # para que apunten a la version super-resuelta.
                os.makedirs(out_dir, exist_ok=True)
                out_json_path = os.path.join(out_dir, filename)
                _copy_metadata_with_updated_paths(in_path, out_json_path, input_root, output_root)
                continue

            if ext not in IMG_EXTENSIONS:
                continue

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)

            img_bgr = cv2.imread(in_path)
            if img_bgr is None:
                print(f"  [aviso] no se pudo leer {in_path}, se omite")
                n_saltadas += 1
                continue

            try:
                sr_img = super_resolve_image(model, img_bgr)
            except Exception as e:
                print(f"  [error] fallo HAT en {in_path}: {e}")
                n_saltadas += 1
                continue

            cv2.imwrite(out_path, sr_img)
            n_procesadas += 1

            if n_procesadas % 50 == 0:
                print(f"  ... {n_procesadas} imagenes procesadas")

    print(f"\nListo. {n_procesadas} imagenes super-resueltas, {n_saltadas} omitidas.")
    print(f"Salida en: {output_root}")


def _copy_metadata_with_updated_paths(in_json_path, out_json_path, input_root, output_root):
    import json
    with open(in_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if "crop_path" in entry:
            entry["crop_path"] = entry["crop_path"].replace(input_root, output_root)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("Cargando modelo HAT...")
    model = load_hat_model()
    print(f"Modelo cargado en '{DEVICE}'. Procesando '{INPUT_ROOT}' -> '{OUTPUT_ROOT}'")
    process_tree(model)


if __name__ == "__main__":
    main()