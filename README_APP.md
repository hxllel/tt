# Prototipo TT 2027-A048: guía de instalación y uso

## Instalación

**Windows (esta laptop, CPU):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

**Linux con GPU NVIDIA:**
```bash
bash scripts/setup_linux.sh              # añade --tensorrt para exportar YOLO a TensorRT
```

Los scripts hacen lo siguiente:
1. `conda init`.
2. Crean el entorno `tt-a048` con conda-forge. No usan los canales `defaults`, que exigen aceptar los ToS de Anaconda.
3. Instalan torch CPU o CUDA 12.4 según `nvidia-smi`, luego `requirements.txt` y `ml-fastvlm` (`--no-deps`, sin coremltools).
4. Descargan los modelos con `download_models.py`: YOLO26 n/s, Real-ESRGAN compacto, FastVLM 0.5B stage3, SigLIP2 y multilingual-e5-small.

Para descargar más checkpoints de FastVLM: `python download_models.py --vlm 1.5b 7b`.

## Uso
```bash
conda activate tt-a048
python app.py                    # autodetecta GPU; abre http://127.0.0.1:7860
python app.py --dispositivo cpu
```
En un servidor remoto usa `python app.py --no-abrir` y `ssh -L 7860:localhost:7860 usuario@host`. **Nunca** `share=True`: los videos son datos personales.

| Pestaña | Qué hace |
|---|---|
| 1. Procesar video | YOLO26 + ByteTrack → mejores recortes por persona → superresolución → embeddings CLIP (y opcionalmente descripciones FastVLM). Registra procedencia y consentimiento (RF-17). |
| 2. Buscar | Consulta con el texto de la alerta. Muestra la galería de coincidencias, la escena con la caja y el contexto de ±3 s. |
| 3. Métricas YOLO | (a) dataset anotado → P/R/mAP y % de personas detectadas; (b) MOT → MOTA/IDF1; (c) estadísticas sin anotaciones. |
| 4. Sistema | GPU y VRAM detectadas, perfil elegido, ajustes y exportación a TensorRT/OpenVINO. |

### Modos de búsqueda
- **clip**: coseno entre el embedding CLIP del recorte y el texto de la alerta (plantilla completa + una por prenda). Es el modo adecuado para miles de personas.
- **hibrido**: CLIP ordena a todas las personas, FastVLM describe solo a los N mejores candidatos, y el orden final combina CLIP con similitud de texto.
- **texto**: coseno e5 entre la descripción de FastVLM y la alerta. Requiere generar descripciones.
- **lexico**: coincidencia de prendas y colores normalizados con sinónimos (línea base de TT1).

## Línea de comandos
```bash
python extract_bounding_boxes.py videos/clip.mp4 --intervalo 5 --modelo yolo26s
python improve_quality.py clip
python search_fastvlm.py clip --describir
python search_fastvlm.py clip --consulta "sudadera gris, pantalón negro" --modo hibrido
python -m pipeline.metrics_yolo stats --video clip
python -m pipeline.metrics_yolo dataset --data datos/personas.yaml
python -m pipeline.metrics_yolo mot --video videos/clip.mp4 --gt gt.txt
python bench_latency.py --device cuda --batches 1 8 16 --video videos/clip.mp4
python -m pytest tests
```

## Selección automática por VRAM (`pipeline/hardware.py`)
| VRAM libre | FastVLM | YOLO | SR | Lote VLM |
|---|---|---|---|---|
| < 4 GB | 0.5B fp16 | 26n | sí | 4 |
| 4–10 GB | 1.5B fp16 | 26s | sí | 8 |
| 10–20 GB | 1.5B fp16 | 26m | sí | 16 |
| ≥ 20 GB | 7B fp16 | 26m | sí | 16 |
| CPU | 0.5B fp32 | 26n | sí (compacta) | 1 |

Si el checkpoint recomendado no existe, se usa el disponible más cercano. Si hay OOM, el lote se divide a la mitad automáticamente.

## Videos con miles de personas
- Usa el modo `clip` o `hibrido`. FastVLM describe a una persona por lote de generación, mientras que CLIP procesa cientos de recortes por segundo en GPU.
- Usa `yolo26s`/`yolo26m` con `imgsz` 1280 si hay personas lejanas. Exporta a TensorRT desde la pestaña Sistema.
- Solo se guardan los `top_k_por_track` recortes por persona, y los tracks se escriben a disco al cerrarse, así que la memoria se mantiene acotada.
- Reescribir en C no aporta: la inferencia ya corre en kernels C++/CUDA (cuDNN, cuBLAS, TensorRT). El cuello de botella eran recargas del modelo, sincronizaciones GPU→CPU por caja y E/S síncrona, y eso ya está corregido.

## Cambios que conviene reflejar en la documentación del TT
- **RNF-13 / Sec. 5.1**: el modo CLIP indexa embeddings de *apariencia de cuerpo completo* (no faciales), almacenados solo localmente.
- **RF-06**: los recortes se organizan por persona (track), no por frame.
- **RF-15**: un solo entorno conda; las etapas siguen comunicándose por archivos (RNF-09/10).
- **"1024×1024"**: es la resolución de entrada del codificador FastViTHD, no la del video.
- **Superresolución**: se aplica solo al mejor recorte pequeño de cada persona. Hay que evaluar con la Capa 2 si mejora o empeora la exactitud de atributos, porque la SR puede inventar texturas.
- **Qwen vs Vicuna**: Apple solo publicó FastVLM con Qwen2. Propuesta: comparar tamaños 0.5B/1.5B/7B y comparar CLIP contra FastVLM.
