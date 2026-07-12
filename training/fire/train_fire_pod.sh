#!/bin/bash
# Entrena el detector de humo/fuego (YOLOX-s) en un pod RunPod (GPU).
# Descarga D-Fire, lo convierte a COCO, entrena y exporta a ONNX.
#
# Requiere UN metodo de descarga del dataset (elegir con DFIRE_SRC):
#   - kaggle : usa la version "ready to use" de Kaggle. Necesita ~/.kaggle/kaggle.json
#              (token de la API de Kaggle; lo genera el usuario en su cuenta).
#   - url    : baja un .zip desde una URL directa que le pases en DFIRE_URL
#              (por ej. un mirror propio o un release de GitHub).
#
# Uso en el pod:
#   cd /workspace
#   DFIRE_SRC=kaggle bash train_fire_pod.sh
set -e
cd /workspace

# ---- 1) YOLOX (Apache 2.0) + pesos base ----
if [ ! -d YOLOX ]; then
  git clone -q https://github.com/Megvii-BaseDetection/YOLOX.git
fi
cd YOLOX
# Deps de entrenamiento explicitas (puras/wheels). El `requirements.txt` de YOLOX
# incluye onnx-simplifier, que necesita cmake para compilar y, si falla, aborta TODO
# el install dejando el entorno a medias. Por eso instalamos lo necesario a mano.
apt-get install -y -q cmake >/dev/null 2>&1 || true
pip install -q loguru thop tabulate tqdm psutil tensorboard ninja \
  opencv-python-headless pycocotools onnx onnxruntime kaggle
pip install -q -e . 2>&1 | tail -1 || true   # editable install opcional; si falla usamos PYTHONPATH
mkdir -p weights
[ -f weights/yolox_s.pth ] || wget -q \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth \
  -O weights/yolox_s.pth

# ---- 2) Descargar D-Fire ----
RAW=/workspace/dfire_raw
if [ ! -d "$RAW" ]; then
  mkdir -p "$RAW"
  case "${DFIRE_SRC:-kaggle}" in
    kaggle)
      # Version "ready to use" (formato YOLO, splits train/valid/test).
      kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo -p "$RAW" --unzip ;;
    url)
      wget -q "$DFIRE_URL" -O "$RAW/dfire.zip" && (cd "$RAW" && unzip -q dfire.zip) ;;
    *)
      echo "DFIRE_SRC invalido"; exit 1 ;;
  esac
fi
echo "=== estructura del dataset descargado (revisar el data.yaml para el orden de clases) ==="
find "$RAW" -maxdepth 2 -type d | head
find "$RAW" -iname "*.yaml" -exec echo "--- {} ---" \; -exec cat {} \; 2>/dev/null | head -20

# ---- 3) Convertir YOLO -> COCO ----
# Ajustar TRAIN_IMG/TRAIN_LBL/VAL_* segun la estructura real que imprima el paso 2.
# (La version de Kaggle suele traer train/images, train/labels, valid/images, valid/labels.)
DS=/workspace/YOLOX/datasets/fire
mkdir -p "$DS/train2017" "$DS/val2017" "$DS/annotations"
TRAIN_IMG=$(find "$RAW" -type d -path "*train*" -name images | head -1)
TRAIN_LBL=$(find "$RAW" -type d -path "*train*" -name labels | head -1)
VAL_IMG=$(find "$RAW" -type d \( -path "*valid*" -o -path "*val*" -o -path "*test*" \) -name images | head -1)
VAL_LBL=$(find "$RAW" -type d \( -path "*valid*" -o -path "*val*" -o -path "*test*" \) -name labels | head -1)
echo "train imgs: $TRAIN_IMG | val imgs: $VAL_IMG"
cp "$TRAIN_IMG"/* "$DS/train2017/" 2>/dev/null || true
cp "$VAL_IMG"/*   "$DS/val2017/"   2>/dev/null || true
python /workspace/yolo_to_coco.py --images "$TRAIN_IMG" --labels "$TRAIN_LBL" \
  --out "$DS/annotations/instances_train2017.json" --names smoke fire
python /workspace/yolo_to_coco.py --images "$VAL_IMG" --labels "$VAL_LBL" \
  --out "$DS/annotations/instances_val2017.json" --names smoke fire

# ---- 4) Entrenar ----
# TRAIN_GPU fija en que GPU entrena. Con 2 GPUs, TRAIN_GPU=0 deja la GPU 1 para
# servir inferencia en paralelo. Para usar las 2 GPUs y terminar mas rapido (pero
# sin GPU libre para inferencia): sin fijar CUDA_VISIBLE_DEVICES y -d 2 -b 32.
cp /workspace/yolox_fire_exp.py /workspace/YOLOX/yolox_fire_exp.py
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}"
# PYTHONPATH=raiz de YOLOX: 'python tools/train.py' pone tools/ en sys.path, no la
# raiz, asi que 'import yolox' falla sin esto (el editable install no siempre entra).
PYTHONPATH="$(pwd)" python tools/train.py -f yolox_fire_exp.py -d 1 -b 16 --fp16 -c weights/yolox_s.pth

# ---- 5) Exportar a ONNX (entrada dinamica, para inferir a 640 o 1280) ----
BEST=YOLOX_outputs/yolox_fire/best_ckpt.pth
PYTHONPATH="$(pwd)" python tools/export_onnx.py -f yolox_fire_exp.py -c "$BEST" \
  --output-name /workspace/yolox_fire.onnx --dynamic --decode_in_inference
echo "==== LISTO: /workspace/yolox_fire.onnx ===="
