"""Filtra un export COCO de Roboflow a UNA sola clase y lo deja en el layout de YOLOX.

Roboflow exporta en formato COCO como:
    <src>/train/_annotations.coco.json + imagenes
    <src>/valid/_annotations.coco.json + imagenes
Este script se queda con las cajas de una clase (ej. "Boots"), las remapea a una
unica categoria, y arma:
    <dst>/annotations/instances_train2017.json  (+ val2017)
    <dst>/train2017  -> symlink a <src>/train   (+ val2017)
Las imagenes sin la clase quedan igual = negativos (fondo), que ayudan al detector.

Uso:
    python roboflow_coco_filter.py <src_dir> <dst_dir> <keep_class> <out_name>
    ej: python roboflow_coco_filter.py /dev/shm/boots_raw \
        /workspace/YOLOX/datasets/boots Boots safety-boot
"""
import json
import os
import shutil
import sys

SRC, DST, KEEP, OUTNAME = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(os.path.join(DST, "annotations"), exist_ok=True)


def process(split_dir, out_json, img_link):
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        print("no existe", ann_path); return
    coco = json.load(open(ann_path))
    ids = [c["id"] for c in coco["categories"] if c["name"].lower() == KEEP.lower()]
    if not ids:
        print("clase", KEEP, "no esta. Clases:", [c["name"] for c in coco["categories"]]); return
    kid = ids[0]
    anns = [a for a in coco["annotations"] if a["category_id"] == kid]
    for a in anns:
        a["category_id"] = 1
    json.dump({"images": coco["images"], "annotations": anns,
               "categories": [{"id": 1, "name": OUTNAME, "supercategory": "none"}]},
              open(out_json, "w"))
    if os.path.islink(img_link):
        os.remove(img_link)
    elif os.path.exists(img_link):
        shutil.rmtree(img_link, ignore_errors=True)
    os.symlink(os.path.abspath(split_dir), img_link)
    print(f"{out_json}: {len(coco['images'])} imgs, {len(anns)} cajas de {KEEP}")


process(os.path.join(SRC, "train"), os.path.join(DST, "annotations", "instances_train2017.json"),
        os.path.join(DST, "train2017"))
process(os.path.join(SRC, "valid"), os.path.join(DST, "annotations", "instances_val2017.json"),
        os.path.join(DST, "val2017"))
