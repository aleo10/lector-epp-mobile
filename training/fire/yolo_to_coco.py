"""Convierte un split en formato YOLO (D-Fire) al formato COCO que espera YOLOX.

D-Fire trae, por cada imagen `foo.jpg`, un `foo.txt` con una linea por caja:
    <class_id> <cx> <cy> <w> <h>       (todo normalizado 0..1, centro+tamaño)
YOLOX entrena sobre COCO: un unico JSON con images/annotations/categories y las
cajas en pixeles [x, y, w, h] (esquina sup-izq + tamaño).

Uso:
    python yolo_to_coco.py --images DIR_IMAGENES --labels DIR_LABELS \
        --out annotations/instances_train2017.json --names smoke fire

⚠️ El orden de --names define el mapeo id->nombre. Verificá contra el data.yaml del
   D-Fire descargado (el indice 0/1 de smoke/fire tiene que coincidir). El default
   asume 0=smoke, 1=fire; si el dataset viene al reves, invertí --names.
"""
import argparse
import json
import os

from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def convert(images_dir, labels_dir, out_json, class_names):
    categories = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(class_names)]
    images, annotations = [], []
    ann_id, img_id = 1, 1
    skipped = 0

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS))
    for fn in files:
        path = os.path.join(images_dir, fn)
        try:
            with Image.open(path) as im:
                W, H = im.size
        except Exception:
            skipped += 1
            continue
        images.append({"id": img_id, "file_name": fn, "width": W, "height": H})

        label_path = os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    c, cx, cy, w, h = (float(p) for p in parts)
                    bw, bh = w * W, h * H
                    x, y = cx * W - bw / 2, cy * H - bh / 2
                    # clamp por si alguna caja se pasa del borde
                    x, y = max(0.0, x), max(0.0, y)
                    bw, bh = min(bw, W - x), min(bh, H - y)
                    if bw <= 0 or bh <= 0:
                        continue
                    annotations.append({
                        "id": ann_id, "image_id": img_id, "category_id": int(c),
                        "bbox": [x, y, bw, bh], "area": bw * bh,
                        "iscrowd": 0, "segmentation": [],
                    })
                    ann_id += 1
        img_id += 1

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"{out_json}: {len(images)} imagenes, {len(annotations)} cajas, "
          f"{skipped} imagenes ilegibles, clases={class_names}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="+", default=["smoke", "fire"])
    a = ap.parse_args()
    convert(a.images, a.labels, a.out, a.names)
