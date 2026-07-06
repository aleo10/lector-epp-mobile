#!/bin/bash
# Setup completo del backend de inferencia en un pod RunPod (GPU).
# Uso en el pod:
#   cd /workspace && \
#   wget -qO- https://raw.githubusercontent.com/aleo10/lector-epp-mobile/main/server/deploy_pod.sh | bash
#
# Deja el servidor WebSocket corriendo en el puerto 8765 usando la GPU.
set -e
cd /workspace

# 1) Codigo + modelo (idempotente; actualiza si ya existe)
if [ -d app ]; then
  git -C app pull -q || true
else
  git clone -q https://github.com/aleo10/lector-epp-mobile.git app
fi
cd app/server
[ -f yolox_ppe_1280.onnx ] || wget -q \
  https://github.com/aleo10/lector-epp-mobile/releases/download/modelos-v1/yolox_ppe_1280.onnx \
  -O yolox_ppe_1280.onnx

# 2) Dependencias: onnxruntime-gpu para CUDA 12 + cuDNN 9
pip install -q websockets opencv-python-headless numpy
pip install -q "onnxruntime-gpu==1.19.2" nvidia-cudnn-cu12 2>&1 | tail -1

# 3) Path de cuDNN (para que onnxruntime encuentre libcudnn.so.9)
CUDNN_DIR=$(python3 -c "import os,nvidia.cudnn as c; print(os.path.join(os.path.dirname(c.__file__),'lib'))")
export LD_LIBRARY_PATH="$CUDNN_DIR:$LD_LIBRARY_PATH"

# 4) Arrancar servidor (import torch primero => inicializa contexto CUDA)
pkill -f infer_server 2>/dev/null || true
sleep 2
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" nohup python3 infer_server.py > /workspace/server.log 2>&1 &
sleep 15
echo "==== server.log ===="
cat /workspace/server.log
