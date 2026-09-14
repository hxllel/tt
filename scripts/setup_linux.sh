#!/usr/bin/env bash
# Instalacion en Linux con GPU NVIDIA (p. ej. el equipo de Hillel).
#   bash scripts/setup_linux.sh            # CUDA 12.4
#   bash scripts/setup_linux.sh --tensorrt # ademas exporta YOLO a TensorRT
set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda env list | grep -q "^tt-a048 " || conda env create -f environment.yml
conda activate tt-a048

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
else
  echo "nvidia-smi no encontrado: se instala torch CPU"
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
pip install --no-deps -e ./ml-fastvlm
python download_models.py

if [[ "${1:-}" == "--tensorrt" ]]; then
  pip install tensorrt-cu12
  python - <<'EOF'
from pipeline.runner import Motor
m = Motor()
print("Exportado:", m.exportar_detector())
EOF
fi

python -m pipeline.hardware
echo "Listo: conda activate tt-a048 && python app.py --no-abrir   (en tu laptop: ssh -L 7860:localhost:7860 usuario@host)"
