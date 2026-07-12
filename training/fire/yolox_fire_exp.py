# Exp de YOLOX para deteccion de HUMO y FUEGO (deteccion temprana de incendios).
# Dataset: D-Fire (CC0, ~21.5k imagenes) convertido a COCO con yolo_to_coco.py.
# Basado en YOLOX-s (depth=0.33, width=0.50), igual que el modelo de EPP.
#
# 2 categorias (verificar el orden contra el data.yaml del D-Fire descargado):
#   0 smoke      1 fire
#
# Uso en el pod:
#   cd /workspace/YOLOX
#   python tools/train.py -f yolox_fire_exp.py -d 1 -b 16 --fp16 -c weights/yolox_s.pth

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # --- Arquitectura: YOLOX-s ---
        self.depth = 0.33
        self.width = 0.50

        # --- Clases: humo + fuego ---
        self.num_classes = 2

        # --- Dataset (COCO) ---
        self.data_dir = "/workspace/YOLOX/datasets/fire"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --- Entrenamiento ---
        self.max_epoch = 100
        self.data_num_workers = 8
        self.eval_interval = 10
        self.print_interval = 20

        # Las imagenes de D-Fire son ~416px; entrenar a 640 alcanza y sobra. La
        # inferencia despues puede correr a 1280 (el mismo modelo, letterbox) para
        # captar humo lejano, igual que hacemos con el EPP.
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 5

        # Fine-tuning desde pesos COCO de YOLOX-s: warmup corto y lr algo mas bajo.
        self.warmup_epochs = 3
        self.basic_lr_per_img = 0.01 / 64.0
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.exp_name = "yolox_fire"
