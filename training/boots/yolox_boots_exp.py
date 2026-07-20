# Exp de YOLOX para el detector DEDICADO de botines de seguridad (corre en paralelo
# con el modelo de EPP). 2 clases:
#   0 safety-boot  (bota de seguridad -> cumple)
#   1 shoes        (calzado comun / sin bota -> incumple)
#
# Dataset: combinar fuentes PERMISIVAS (CC-BY / Public Domain) de Roboflow con botines
# (ej. "ppe detection public" de Canberk, que tiene safety-boot + shoes). NO usar el
# Construction-PPE de Ultralytics (AGPL). Exportar de Roboflow en formato COCO o YOLO
# y (si YOLO) convertir con ../fire/yolo_to_coco.py --names safety-boot shoes.
#
# ⚠️ Verificar el orden de clases del dataset descargado (data.yaml) y ajustar --names.
#
# Uso en el pod (mismo flujo que el fuego):
#   cd /workspace/YOLOX
#   PYTHONPATH=/workspace/YOLOX python tools/train.py -f yolox_boots_exp.py \
#       -d 1 -b 16 --fp16 -c weights/yolox_s.pth

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # YOLOX-s
        self.depth = 0.33
        self.width = 0.50

        # v1: 1 clase (safety-boot). Data: SiaBar (CC-BY), clase "Boots" = 2430 instancias.
        # Ausencia de botín en una persona -> "verificando" (conservador; pies tapados).
        self.num_classes = 1

        self.data_dir = "/workspace/YOLOX/datasets/boots"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.max_epoch = 80          # dataset chico -> algo mas de epochs que el fuego
        self.data_num_workers = 16   # OJO: no poner = nproc (128 cuelga el arranque)
        self.eval_interval = 10
        self.print_interval = 20

        # Botines son objetos CHICOS (pies) -> conviene entrenar a mayor resolucion
        # que 640 para que aprenda a detectarlos. 896 es un buen compromiso.
        self.input_size = (896, 896)
        self.test_size = (896, 896)
        self.multiscale_range = 5

        # Fine-tuning desde pesos COCO
        self.warmup_epochs = 3
        self.basic_lr_per_img = 0.01 / 64.0
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.exp_name = "yolox_boots"
