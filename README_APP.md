# Prototipo TT 2027-A048 — Guía completa

Búsqueda de personas en grabaciones de video a partir de la descripción de vestimenta de una alerta oficial (Alerta Amber, cédula de búsqueda). Sistema **offline**, por lotes, sobre grabaciones ya almacenadas.

---

## Índice

1. [Qué hace y cómo funciona](#1-qué-hace-y-cómo-funciona)
2. [Requisitos](#2-requisitos)
3. [Instalación desde cero](#3-instalación-desde-cero)
4. [Verificar que quedó bien instalado](#4-verificar-que-quedó-bien-instalado)
5. [Uso de la interfaz gráfica](#5-uso-de-la-interfaz-gráfica)
6. [Uso por línea de comandos](#6-uso-por-línea-de-comandos)
7. [Configuración (`config.yaml`)](#7-configuración-configyaml)
8. [Rendimiento: GPU y CPU](#8-rendimiento-gpu-y-cpu)
9. [Métricas de evaluación](#9-métricas-de-evaluación)
10. [Archivos que genera](#10-archivos-que-genera)
11. [Solución de problemas](#11-solución-de-problemas)
12. [Estructura del proyecto](#12-estructura-del-proyecto)
13. [Privacidad y uso responsable](#13-privacidad-y-uso-responsable)
14. [Cambios pendientes en la documentación del TT](#14-cambios-pendientes-en-la-documentación-del-tt)

---

## 1. Qué hace y cómo funciona

El único modelo que "ve" la imagen es **FastVLM**. No se usa CLIP ni ningún otro modelo de visión.

```
Video
 │
 ├─ Etapa 1 — Extracción
 │    OpenCV lee 1 de cada N frames (no decodifica el resto)
 │    CLAHE solo si el frame está oscuro                        → RF-03
 │    YOLO26 detecta personas + ByteTrack las sigue             → RF-04, RF-05
 │    Por cada persona (track) se guardan sus K mejores recortes
 │    → salidas/<video>/crops/ + metadata.json                  → RF-06, RF-07
 │
 ├─ Etapa 1b — Superresolución (opcional)
 │    Real-ESRGAN sobre el mejor recorte de cada persona, solo si es pequeño
 │    → salidas/<video>/crops_sr/
 │
 ├─ Etapa 2 — Descripción
 │    FastVLM describe UNA vez a cada persona, con un prompt genérico
 │    → salidas/<video>/descripciones.json                      → RF-08, RF-09
 │
 └─ Etapa 3 — Búsqueda (se puede repetir sin reprocesar el video)
      Consulta = texto de la alerta
      · "texto"     → embeddings e5 de la descripción vs la alerta, coseno    → RF-12
      · "lexico"    → coincidencia de prendas y colores normalizados          → RF-11
      · "combinado" → 50/50 de los dos anteriores (el que mejor funciona)
      → ranking de personas con su minuto exacto de aparición                 → RF-13
```

**Por qué el prompt es genérico:** la descripción se genera una sola vez y sirve para cualquier búsqueda posterior. Cambiar la consulta no vuelve a ejecutar FastVLM (RF-09).

**Por qué la descripción sale en inglés:** FastVLM fue entrenado con instrucciones en inglés; pidiéndole español o JSON, la variante 0.5B ignora la instrucción y alucina. El modelo de embeddings (`multilingual-e5-small`) es multilingüe, así que empareja una alerta escrita en español con una descripción en inglés sin problema. El modo léxico también traduce sinónimos de los dos idiomas.

**Por qué los embeddings son solo de texto:** así se cumple RNF-13 — no se almacena ninguna representación de la imagen ni biométrica, únicamente el vector del texto generado.

**Por qué hay un modo combinado:** medido sobre un video de prueba, cada modo falla distinto. El semántico tolera que la redacción cambie ("chamarra" ↔ "jacket") pero no amarra bien qué color va con qué prenda: puntúa casi igual a alguien de pantalón negro que a uno de pantalón blanco. El léxico sí amarra color y prenda, pero no entiende sinónimos fuera del diccionario. Combinados al 50 % (tras normalizar cada escala), la persona correcta sube al primer puesto. Los dos modos puros se conservan porque el protocolo pide comparar la línea base léxica de TT1 contra la recuperación semántica de TT2.

---

## 2. Requisitos

| | Mínimo | Recomendado |
|---|---|---|
| Sistema | Windows 10/11, Ubuntu 20.04+ o macOS | Ubuntu 22.04 |
| Python | 3.10 (lo instala conda) | 3.10 |
| RAM | 8 GB | 16 GB |
| GPU | ninguna (funciona en CPU) | NVIDIA ≥ 8 GB VRAM, driver con CUDA 12.x |
| Disco | ~12 GB | 20 GB |

Antes de empezar necesitas **Miniconda** y **git**.

- Miniconda: https://www.anaconda.com/download/success (sección Miniconda)
- Comprueba la GPU con `nvidia-smi`. Si el comando no existe, no hay GPU NVIDIA utilizable y todo correrá en CPU.

Qué ocupa el disco: FastVLM 0.5B ≈ 1.5 GB, FastVLM 1.5B ≈ 3.6 GB, PyTorch con CUDA ≈ 3 GB, el resto de modelos < 0.5 GB.

---

## 3. Instalación desde cero

### 3.1 Vía rápida (scripts)

**Linux / macOS**
```bash
git clone <url-del-repo> tt && cd tt && bash scripts/setup_linux.sh
```
Añade `--tensorrt` al final para exportar además YOLO a TensorRT (solo GPU NVIDIA).

**Windows (PowerShell)**
```powershell
git clone <url-del-repo> tt; cd tt; powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

Los scripts ejecutan exactamente los pasos de la sección siguiente. Si fallan a medio camino, continúa a mano desde el paso que falló.

### 3.2 Paso a paso (manual)

**Paso 1 — Clonar el repositorio**
```bash
git clone <url-del-repo> tt
cd tt
```
El repositorio debe incluir la carpeta `ml-fastvlm/` (el código de Apple). Si no está:
```bash
git clone https://github.com/apple/ml-fastvlm.git ml-fastvlm
```

**Paso 2 — Habilitar `conda` en tu terminal**
```bash
conda init bash
```
En Windows PowerShell es `conda init powershell`. **Cierra la terminal y abre una nueva** para que surta efecto. Si `conda` no se encuentra, usa la ruta completa: `~/miniconda3/bin/conda` en Linux o `$env:USERPROFILE\miniconda3\Scripts\conda.exe` en Windows.

**Paso 3 — Crear el entorno**
```bash
conda env create -f environment.yml
```
```bash
conda activate tt-a048
```
> `environment.yml` usa **solo conda-forge**, a propósito. Los canales `defaults` de Anaconda exigen aceptar sus Términos de Servicio y, si no se han aceptado, `conda create` falla sin explicar por qué. Si prefieres los canales por defecto, acéptalos tú con `conda tos accept`.

**Paso 4 — Instalar PyTorch**

Con GPU NVIDIA (comprueba antes con `nvidia-smi`):
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```
Sin GPU:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```
> Instala PyTorch **antes** que el resto: así pip no baja una versión genérica que luego habría que reemplazar.

**Paso 5 — Instalar el resto de dependencias**
```bash
pip install -r requirements.txt
```
```bash
pip install --no-deps -e ./ml-fastvlm
```
> `--no-deps` es obligatorio: el `pyproject.toml` de Apple fija versiones viejas (gradio 5.11, timm 1.0.15) y `coremltools`, que solo sirve en macOS. Verás un aviso de pip diciendo que `llava` tiene dependencias insatisfechas; es esperado y no afecta.

**Paso 6 — Descargar los modelos**
```bash
python download_models.py
```
Baja a `models/` y `ml-fastvlm/checkpoints/`:

| Modelo | Tamaño | Para qué |
|---|---|---|
| `yolo26n.pt`, `yolo26s.pt` | 25 MB | detección de personas |
| `realesr-general-x4v3.pth` | 5 MB | superresolución de recortes pequeños |
| `llava-fastvithd_0.5b_stage3` | 1.5 GB | FastVLM, descripción |
| `multilingual-e5-small` | 0.5 GB | embeddings de texto para la búsqueda |

Opciones:
```bash
python download_models.py --vlm 0.5b 1.5b     # también el modelo mediano (3.6 GB, mejor calidad)
python download_models.py --yolo yolo26m      # detector más grande, para escenas con gente lejana
python download_models.py --hat               # instrucciones para usar HAT en vez de Real-ESRGAN
```
Con GPU de 8 GB o más, baja también el `1.5b`: el sistema lo elegirá solo.

Si la descarga de FastVLM se corta, bórrala y reinténtala:
```bash
rm -rf ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3*
```

**Paso 7 — Inicializar el entorno**
```bash
python init_entorno.py --inicializar
```
Detecta el hardware, crea `models/`, `videos/` y `salidas/`, y escribe un **`config.local.yaml`** con los valores concretos de esta máquina (dispositivo, detector, `imgsz`, tamaño de FastVLM, lote, superresolución). Ese archivo sobreescribe lo que haga falta de `config.yaml` y **no se versiona**, así que cada equipo tiene el suyo: la laptop sin GPU y el servidor con RTX 4090 comparten el mismo repositorio sin pisarse. Para volver a los valores automáticos, bórralo.

**Paso 8 (opcional, solo GPU) — Acelerar YOLO con TensorRT**
```bash
pip install tensorrt-cu12
```
```bash
python -c "from pipeline.runner import Motor; print(Motor().exportar_detector())"
```
Tarda varios minutos y genera un `.engine` que el sistema usará automáticamente. También se puede hacer desde la pestaña **Sistema** de la interfaz. El `.engine` sirve solo para esa GPU y esa versión de TensorRT: si cambias de máquina, vuelve a exportarlo.

---

## 4. Verificar que quedó bien instalado

**1. ¿Está todo en su sitio?**
```bash
python init_entorno.py
```
Revisa Python, el entorno conda, CPU, RAM, disco, GPU y VRAM, cada paquete, el código de Apple, cada modelo y las carpetas, y termina con el perfil que usará este equipo. Cada línea `[FALTA]` trae el comando exacto que la arregla. Devuelve código de salida 1 si falta algo crítico, así que sirve dentro de un script o en CI.

```
[ok]    Python                  3.10.21 (.../envs/tt-a048/python.exe)
[ok]    GPU                     NVIDIA GeForce RTX 4090 | 23.1 GB libres de 24.0 GB | CUDA 12.4
[ok]    Checkpoints FastVLM     0.5b, 1.5b
[FALTA] Modelo yolo26n.pt       no descargado (detector ligero)
[ok]    Perfil para este equipo cuda | FastVLM 1.5b | yolo26m | lote 16 | fp16 | SR si
```

| Comando | Qué hace |
|---|---|
| `python init_entorno.py` | solo diagnostica, no toca nada |
| `python init_entorno.py --inicializar` | crea las carpetas y escribe `config.local.yaml` |
| `python init_entorno.py --descargar` | descarga los modelos que falten, con el tamaño de FastVLM adecuado a esta GPU |
| `python init_entorno.py --todo` | las dos cosas |
| `python init_entorno.py --json` | el mismo informe en JSON, para scripts |

Está escrito para poder ejecutarse **antes** de instalar nada: solo importa la biblioteca estándar, así que en un equipo recién clonado te dice qué falta en lugar de fallar con un `ImportError`.

**1b. ¿El perfil elegido es el esperado?**
```bash
python -m pipeline.hardware
```
Imprime la detección cruda: `device='cuda'` con el nombre de tu GPU, o `device='cpu'`.

**2. ¿Pasan las pruebas?**
```bash
python -m pytest tests -q
```
Esperado: `16 passed`.

**3. Prueba completa con un video propio**
```bash
python extract_bounding_boxes.py videos/mi_video.mp4
```
```bash
python search_fastvlm.py mi_video --describir --consulta "playera roja y pantalón negro"
```

**4. Levantar la interfaz**
```bash
python app.py
```
Debe abrirse http://127.0.0.1:7860 en el navegador.

---

## 5. Uso de la interfaz gráfica

```bash
conda activate tt-a048
```
```bash
python app.py
```

Opciones de arranque:

| Comando | Para qué |
|---|---|
| `python app.py --dispositivo cpu` | forzar CPU aunque haya GPU |
| `python app.py --dispositivo cuda` | fallar de inmediato si no hay GPU, en vez de caer en CPU |
| `python app.py --port 7870` | otro puerto |
| `python app.py --no-abrir` | no abrir el navegador (servidores) |

**Desde otra máquina (GPU remota):** en el servidor `python app.py --no-abrir`, y en tu equipo un túnel SSH:
```bash
ssh -L 7860:localhost:7860 usuario@servidor
```
Luego abre http://127.0.0.1:7860. **Nunca uses `share=True`**: los videos contienen datos personales (RNF-14, RNF-15).

### Pestaña 1 — Procesar video

1. Sube los videos, o mejor, escribe sus rutas en disco, una por línea (evita copiar archivos grandes).
2. Indica la **procedencia** y marca el **consentimiento informado** si son voluntarios (RF-17; el sistema no deja continuar sin marcarlo).
3. Deja activado **"Describir a cada persona con FastVLM"**. Sin descripciones no se puede buscar.
4. **Procesar**. La barra muestra cada etapa y al terminar aparece una tabla con tiempos y número de personas.

Reprocesar un video **borra y regenera** su carpeta en `salidas/`.

### Pestaña 2 — Buscar

1. Elige el video procesado (el botón ↻ refresca la lista).
2. Pega el texto de la alerta tal cual: *"Vestía playera roja, pantalón de mezclilla azul, tenis blancos y mochila negra"*.
3. Elige el modo:
   - **Combinada** — recomendada. Promedia los dos siguientes tras normalizarlos.
   - **Semántica (e5)** — tolera que la redacción no coincida: "chamarra" encuentra "jacket", "sudadera gris" encuentra "gray hoodie". La consulta se traduce antes al inglés con el diccionario de atributos, porque FastVLM describe en ese idioma. Flojea al asociar cada color con su prenda.
   - **Léxica por atributos** — línea base de TT1. Cuenta cuántas prendas de la consulta aparecen en la descripción: prenda con el color correcto vale 1, con otro color 0.5. Es instantánea, explicable y precisa en la asociación color–prenda, pero solo conoce los sinónimos de su diccionario.
4. **Buscar**. Abajo salen la galería de coincidencias, la tabla con minuto de aparición y el intervalo en cámara. Al hacer clic en una coincidencia se ve el frame completo con la persona marcada y el contexto de ±3 s.

El botón **"Generar descripciones FastVLM de todo el video"** sirve si procesaste sin describir. Solo describe lo que falta, así que es seguro repetirlo.

### Pestaña 3 — Métricas YOLO

Tres modos, según lo que tengas anotado. Detalles en la [sección 9](#9-métricas-de-evaluación).

### Pestaña 4 — Sistema

Muestra GPU, VRAM libre y el perfil elegido. Permite cambiar dispositivo, detector, `imgsz`, intervalo de frames, superresolución, tamaño de FastVLM, resolución del codificador y formato de salida — y exportar YOLO a TensorRT u OpenVINO.

Los cambios aplican desde el siguiente procesamiento y **no se guardan**. Para fijarlos en este equipo, edita `config.local.yaml` (o regenéralo con `python init_entorno.py --inicializar`); para cambiarlos en todo el proyecto, `config.yaml`.

---

## 6. Uso por línea de comandos

Útil para lotes largos, para correr por SSH sin interfaz y para los experimentos del reporte.

**Etapa 1 — detección y tracking**
```bash
python extract_bounding_boxes.py videos/clip1.mp4 videos/clip2.mp4
```
```bash
python extract_bounding_boxes.py videos/clip.mp4 --intervalo 5 --modelo yolo26m --imgsz 1280 --dispositivo cuda
```

**Etapa 1b — superresolución**
```bash
python improve_quality.py clip1
```
```bash
python improve_quality.py clip1 --modelo HAT_SRx4_ImageNet-pretrain.pth --max-alto 200
```

**Etapa 2 y 3 — describir y buscar**
```bash
python search_fastvlm.py clip1 --describir
```
```bash
python search_fastvlm.py clip1 --consulta "sudadera gris y pantalón negro" --top-k 10
```
```bash
python search_fastvlm.py clip1 --consulta "sudadera gris y pantalón negro" --modo lexico
```
Modos: `combinado` (por defecto), `texto`, `lexico`. Para el reporte comparativo de TT, corre los tres sobre los mismos casos.

**Benchmark de latencia (Capa 4 del Plan de Validación)**
```bash
python bench_latency.py --device cuda --batches 1 8 16 --resoluciones 1024 768 512 --video videos/clip.mp4
```
Mide TTFT, tokens/s, recortes/s y FPS de YOLO; repite cada medición 20 veces y guarda media ± desviación estándar en un CSV dentro de `salidas/bench/`, junto con el hardware y las versiones usadas.

**Métricas de detección**
```bash
python -m pipeline.metrics_yolo stats --video clip1
```

---

## 7. Configuración (`config.yaml`)

Todo el comportamiento por defecto está en [config.yaml](config.yaml). `auto` significa "decidido según el hardware".

### `deteccion`
| Clave | Por defecto | Qué hace |
|---|---|---|
| `modelo` | `auto` | `yolo26n`/`s`/`m`/`l`. Más grande = detecta gente más lejana, pero más lento |
| `motor` | `auto` | usa `.engine` (TensorRT) u OpenVINO si existen |
| `imgsz` | `auto` | 960 en GPU, 640 en CPU. Sube a 1280 si las personas se ven muy pequeñas |
| `intervalo_frames` | `10` | procesa 1 de cada N frames. Bajarlo mejora el tracking y multiplica el costo |
| `conf_tracker` | `0.1` | **no lo subas**: ByteTrack necesita las detecciones débiles para asociar |
| `conf_recorte` | `0.4` | confianza mínima para guardar un recorte |
| `top_k_por_track` | `3` | recortes guardados por persona |
| `min_detecciones_track` | `3` | descarta tracks fugaces (ruido) |
| `min_alto_px` | `48` | descarta personas demasiado pequeñas |
| `umbral_baja_luz` | `60` | luminancia media bajo la cual se aplica CLAHE |
| `frames_para_cerrar_track` | `60` | cuándo se da por terminada una trayectoria y se escribe a disco |

### `superres`
| Clave | Por defecto | Qué hace |
|---|---|---|
| `activar` | `auto` | `true`/`false` para forzarlo |
| `modelo` | `realesr-general-x4v3.pth` | cualquier checkpoint dentro de `models/` que soporte spandrel (Real-ESRGAN, HAT, SwinIR) |
| `max_alto_px` | `160` | solo mejora el recorte si es más bajo que esto |

> La superresolución **inventa** textura. Antes de darla por buena hay que medir con la Capa 2 del Plan de Validación si sube o baja la exactitud de atributos.

### `vlm`
| Clave | Por defecto | Qué hace |
|---|---|---|
| `modelo` | `auto` | `0.5b`, `1.5b` o `7b` |
| `resolucion` | `1024` | entrada del codificador FastViTHD. 512 da 64 tokens visuales en vez de 256 y es mucho más rápido |
| `max_new_tokens` | `64` | suficiente para una frase de vestimenta |
| `batch` | `auto` | recortes por llamada. Con OOM se reduce solo a la mitad |
| `cpu_int8` | `false` | cuantización int8 en CPU. Probada en Ryzen 4600H: degrada el texto y **no** acelera |
| `formato` | `en` | `en` (recomendado), `json` o `libre` (español). Ver sección 1 |

### Dónde va cada cosa

| Archivo | Alcance | ¿Se versiona? |
|---|---|---|
| `config.yaml` | valores por defecto del proyecto, con `auto` donde depende del hardware | sí |
| `config.local.yaml` | valores concretos de ESTA máquina, los escribe `init_entorno.py` | no |
| argumentos de CLI (`--dispositivo`, `--modelo`, …) | solo esa ejecución | — |

Se aplican en ese orden: lo último gana.

### `busqueda`
| Clave | Por defecto | Qué hace |
|---|---|---|
| `texto_modelo` | `intfloat/multilingual-e5-small` | embeddings de texto |
| `top_k` | `20` | resultados devueltos |

Los sinónimos de prendas y colores, y su traducción al inglés, están en los diccionarios `_SINONIMOS` y `EN_CANON` de [pipeline/search.py](pipeline/search.py). Añadir un término regional ("chamarra de mezclilla", "huaraches") es agregar una palabra a la lista.

---

## 8. Rendimiento: GPU y CPU

### Perfil automático según la VRAM libre

`pipeline/hardware.py` mide la VRAM **libre** al arrancar y elige:

| VRAM libre | FastVLM | YOLO | Lote | Precisión |
|---|---|---|---|---|
| < 4 GB | 0.5B | yolo26n | 4 | fp16 |
| 4–10 GB | 1.5B | yolo26s | 8 | fp16 |
| 10–20 GB | 1.5B | yolo26m | 16 | fp16 |
| ≥ 20 GB | 7B | yolo26m | 16 | fp16 |
| CPU | 0.5B | yolo26n | 1 | fp32 |

Si el checkpoint elegido no está descargado, usa el más cercano que sí esté. Si aun así se queda sin memoria, el lote se divide a la mitad y reintenta; el lote final queda anotado en `stats.json`.

### Optimizaciones ya aplicadas

- **FastVLM se carga una sola vez** por ejecución. La versión original lanzaba un `subprocess` por recorte y recargaba el modelo cada vez (≈20–30 s de puro arranque, por persona).
- **Generación por lotes** con padding a la izquierda. Verificado: el texto es idéntico al de lote 1 y es más rápido.
- **Una sola transferencia GPU→CPU por frame** en la detección. Antes había una por caja (`.item()`), y eso es letal en escenas con mucha gente.
- **`grab()` en los frames que se descartan**: no se decodifican ni se copian.
- **Escritura de JPG en un hilo aparte**, y solo de los K mejores recortes de cada persona.
- **Los tracks se cierran y se escriben a disco** al dejar de verse, así que la memoria no crece con la duración del video.
- **Descripción corta y determinista** (greedy, 64 tokens) en vez de 256 tokens con muestreo.

### Videos con mucha gente

El costo dominante es FastVLM: describe a **cada persona única**, no cada frame. Para un video con 1 000 personas distintas, en una RTX 4090 con el 1.5B en lote 16 son del orden de minutos; en CPU no es viable.

Palancas, de mayor a menor impacto:

1. `vlm.resolucion: 512` — menos tokens visuales, mucho más rápido. Mide antes la pérdida de exactitud con la Capa 2.
2. `vlm.modelo: 0.5b` en vez de 1.5B.
3. Subir `intervalo_frames` (menos frames → menos tracks espurios).
4. Subir `min_detecciones_track` y `min_alto_px` para descartar gente de fondo que no se va a poder describir igual.
5. Exportar YOLO a TensorRT.
6. `top_k_por_track: 1` si no necesitas recortes alternativos.

### Por qué no reescribir en C

La inferencia ya corre en kernels C++/CUDA (cuDNN, cuBLAS, TensorRT); Python solo los orquesta. Los cuellos de botella reales eran recargar el modelo, sincronizar la GPU caja por caja y la escritura síncrona de archivos, y los tres ya están resueltos. Donde sí ayuda el código compilado es en los runtimes: TensorRT en GPU y OpenVINO en CPU, ambos ya soportados.

### Medido en una laptop sin GPU (Ryzen 5 4600H, 6 núcleos)

Video de 10 s, 1024×1024, 600 frames, 6 personas:

| Etapa | Tiempo |
|---|---|
| Detección + tracking (60 frames muestreados) | 9.97 s (6.8 fps) |
| Superresolución (6 recortes) | 0.49 s |
| FastVLM 0.5B (6 personas) | ≈150 s (TTFT ≈ 6 s por persona) |
| Búsqueda semántica (ya descrito) | < 5 s |

---

## 9. Métricas de evaluación

Capa 1 del Plan de Validación. Para saber **cuántas personas detecta realmente** hace falta ground truth: sin anotaciones solo se pueden sacar estadísticas descriptivas.

**(a) Con dataset anotado en formato YOLO** (clase 0 = persona; exporta así desde CVAT o Label Studio)
```bash
python -m pipeline.metrics_yolo dataset --data datos/personas.yaml --conf 0.4
```
Devuelve precision, recall, mAP50 y mAP50-95 estándar, y además, al umbral real de operación: **% de personas del ground truth detectadas**, falsos positivos por imagen, error de conteo (MAE) y recall desglosado por tamaño de caja (chica / mediana / grande). Lo último es lo que dice si el detector se pierde a la gente lejana.

**(b) Con ground truth de tracking (formato MOTChallenge)**
```bash
python -m pipeline.metrics_yolo mot --video videos/clip.mp4 --gt gt.txt --intervalo 1
```
Devuelve MOTA, MOTP, IDF1, cambios de identidad y objetos mayormente seguidos. Ojo: el MOTP de `motmetrics` es `1 − IoU medio`, así que **menor es mejor**.

**(c) Sin anotaciones**
```bash
python -m pipeline.metrics_yolo stats --video clip1
```
Detecciones por frame, personas únicas, distribución de longitud de tracks e histograma de confianza, más una gráfica PNG. **Esto no mide exactitud**, solo describe lo que el detector reportó.

Los resultados se guardan en `salidas/metricas/`.

---

## 10. Archivos que genera

```
salidas/<nombre-del-video>/
├── crops/                  recortes JPG, nombrados track_<id>_r<rango>_f<frame>.jpg
├── crops_sr/               los mismos tras superresolución (solo los pequeños)
├── metadata.json           video, fps, tracks y, por recorte: frame, timestamp,
│                           bbox, confianza, calidad, procedencia, consentimiento
├── descripciones.json      por recorte: texto de FastVLM, TTFT, tokens, errores
├── emb_texto.npz           embeddings de las descripciones (caché de búsqueda)
└── stats.json              tiempos y contadores de cada etapa

config.local.yaml           ajustes de esta máquina (no se versiona)
salidas/metricas/           resultados de métricas YOLO (CSV + JSON)
salidas/bench/              CSV de bench_latency.py
models/                     pesos descargados
videos/                     videos subidos desde la interfaz
```

Todo es inspeccionable de forma independiente (RNF-09, RNF-10). Nada de esto se sube a git: `.gitignore` excluye videos, salidas y pesos.

---

## 11. Solución de problemas

**Lo primero ante cualquier duda**
```bash
python init_entorno.py
```
Dice exactamente qué falta y con qué comando se arregla. Abajo, los casos concretos.

**`conda: command not found`**
No se ejecutó `conda init`, o no se reabrió la terminal. Usa la ruta completa: `~/miniconda3/bin/conda`.

**`CondaToSNonInteractiveError` al crear el entorno**
Son los Términos de Servicio de los canales de Anaconda. `environment.yml` ya evita el problema usando solo conda-forge; si creaste el entorno a mano, añade `-c conda-forge --override-channels` o ejecuta `conda tos accept`.

**`IndexError: list index out of range` en `helpers.py`, al procesar**
Versión vieja de gradio combinada con evaluar `gr.Progress` como booleano. Corregido en el código; actualiza con `pip install -U "gradio>=5.35,<6"`.

**`TypeError: argument of type 'bool' is not iterable` al abrir la página**
gradio 5.11 con pydantic ≥ 2.11. Mismo arreglo: `pip install -U "gradio>=5.35,<6"`.

**`ModuleNotFoundError: No module named 'llava'`**
Falta instalar el paquete de Apple: `pip install --no-deps -e ./ml-fastvlm`.

**`No se encontro un checkpoint de FastVLM en ...`**
No se descargó o quedó incompleto: `python download_models.py --vlm 0.5b`. La ruta debe contener un `config.json`.

**`Se forzo dispositivo=cuda pero torch no detecta una GPU CUDA`**
Comprueba `nvidia-smi` y luego:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Si dice `2.6.0+cpu`, instalaste la rueda equivocada: reinstala con el índice `cu124`.

**`CUDA out of memory`**
El sistema ya reduce el lote solo. Si persiste, en la pestaña Sistema baja el modelo a `0.5b`, la resolución a 512, o fija `vlm.batch` a un número pequeño en `config.yaml`.

**`Este video no tiene descripciones de FastVLM`**
Procesaste sin describir. Usa el botón de la pestaña Buscar o `python search_fastvlm.py <video> --describir`.

**`No se reconocio ninguna prenda en la consulta` (modo léxico)**
La consulta no menciona ninguna prenda conocida. Usa el modo semántico o el combinado, o añade el término al diccionario `_SINONIMOS` de [pipeline/search.py](pipeline/search.py).

**La descripción sale incoherente o repetitiva**
Revisa que `vlm.formato` sea `en`. En español o en JSON, el modelo 0.5B alucina. Si aun así pasa, el recorte probablemente es demasiado pequeño: sube `min_alto_px` o activa la superresolución.

**El procesamiento es lentísimo**
Comprueba en la pestaña Sistema que dice GPU y no CPU. Si es CPU y tienes GPU, el problema es la instalación de PyTorch (ver arriba).

**No se abre el navegador / "localhost is not accessible"**
Usa `python app.py --no-abrir` y entra a mano a http://127.0.0.1:7860, o cambia de puerto con `--port`.

**El video `.mov` da timestamps raros**
Ya se usa `CAP_PROP_POS_MSEC`, que respeta el FPS variable típico de los celulares. Si el contenedor está dañado, reconvierte: `ffmpeg -i entrada.mov -c:v libx264 salida.mp4`.

---

## 12. Estructura del proyecto

```
app.py                     interfaz Gradio
config.yaml                configuración por defecto
download_models.py         descarga de pesos (idempotente)
init_entorno.py            diagnóstico del equipo e inicialización (solo stdlib)
environment.yml            entorno conda (solo conda-forge)
requirements.txt           dependencias pip (torch se instala aparte)
extract_bounding_boxes.py  CLI de la etapa 1
improve_quality.py         CLI de superresolución
search_fastvlm.py          CLI de descripción y búsqueda
bench_latency.py           benchmark de latencia (Capa 4)
pipeline/
  config.py                carga de config.yaml y rutas
  hardware.py              detección de GPU/VRAM y perfiles
  video_io.py              lectura eficiente de video
  detect_track.py          YOLO26 + ByteTrack + selección de recortes
  superres.py              superresolución con spandrel
  describe.py              FastVLM en proceso, por lotes
  embed.py                 embeddings de texto (e5)
  search.py                normalización de consultas y coincidencia léxica
  runner.py                orquestador, modelos cargados una vez
  metrics_yolo.py          métricas de detección y tracking
scripts/                   instaladores para Linux y Windows
tests/                     pruebas unitarias (pytest)
ml-fastvlm/                repositorio de Apple (sin modificar)
```

---

## 13. Privacidad y uso responsable

- El sistema es **offline**: no se conecta a cámaras en vivo ni sube nada a servicios externos (RNF-14).
- **No usa rostros ni datos biométricos.** Solo compara vestimenta. Los únicos embeddings guardados son de texto (RNF-13).
- Toda coincidencia es una **sugerencia sujeta a verificación humana**, no una identificación positiva (RNF-12). La interfaz lo muestra de forma permanente.
- Las grabaciones de voluntarios requieren **consentimiento informado por escrito**; la interfaz obliga a registrarlo (RF-17).
- No uses `share=True` ni expongas el puerto fuera de `localhost`. Para acceso remoto, túnel SSH.
- Videos, recortes y salidas están excluidos de git. Verifica con `git status` antes de cada commit.

---

## 14. Cambios pendientes en la documentación del TT

Hay que alinear el protocolo con lo que realmente hace el sistema:

- **RF-06** — los recortes se organizan por persona (track), no por frame.
- **RF-15** — hay un solo entorno conda, no dos. Las etapas siguen comunicándose por archivos, que es lo que RNF-09 y RNF-10 realmente piden.
- **"1024×1024"** — es la resolución de entrada del codificador FastViTHD, no la del video. Conviene reformularlo en el objetivo general, RF-01 y RNF-02.
- **Superresolución** — no está en el protocolo y hay que justificarla, aclarando que no sustituye a RF-03 (mejora de iluminación, que se resuelve con CLAHE).
- **Qwen vs Vicuna (Fase 3)** — Apple solo publicó FastVLM con Qwen2; comparar contra Vicuna obliga a reentrenar desde cero y agrava el riesgo R-01. Propuesta: comparar tamaños 0.5B / 1.5B / 7B y, como referencia externa, un VLM estándar (RNF-01, RNF-03 lo exigen de todos modos).
- **R-08** — el riesgo de latencia por `subprocess` ya está mitigado: el modelo se carga una sola vez.
- **Idioma de las descripciones** — conviene documentar que FastVLM responde en inglés y que el emparejamiento con la alerta en español lo resuelve el modelo multilingüe de embeddings.
