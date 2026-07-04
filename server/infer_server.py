"""Servidor de inferencia YOLOX por WebSocket para el modo RunPod (GPU).

El celular (PWA) abre un WebSocket y manda frames JPEG (binario). El servidor
decodifica, corre YOLOX en GPU y responde JSON con las detecciones. Solo devuelve
las clases relevantes de EPP (persona + casco/chaleco/barbijo).

Despliegue en un pod RunPod (GPU):
    pip install -r requirements.txt
    python infer_server.py                    # escucha en 0.0.0.0:8765
Exponer el puerto 8765 en el pod -> el proxy de RunPod da:
    wss://<POD_ID>-8765.proxy.runpod.net
Esa URL es la que se pega en la PWA (modo RunPod).
"""
import asyncio
import json
import cv2
import numpy as np
import onnxruntime as ort
import websockets

MODEL_PATH = "yolox_ppe.onnx"
INPUT_SIZE = 640
CONF_THRES = 0.35
NMS_THRES = 0.45
PORT = 8765

CLASSES = ["construction", "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
           "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest",
           "machinery", "vehicle"]
SHOW = {"Person", "Hardhat", "NO-Hardhat", "Mask", "NO-Mask",
        "Safety Vest", "NO-Safety Vest"}

# GPU si esta disponible, si no CPU.
providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if "CUDAExecutionProvider" in ort.get_available_providers()
             else ["CPUExecutionProvider"])
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name
print(f"Modelo cargado. Providers: {session.get_providers()}")


def preprocess(img):
    h, w = img.shape[:2]
    r = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32)  # CHW, BGR
    return blob, r


def detect(img):
    blob, r = preprocess(img)
    out = session.run(None, {input_name: blob})[0][0]
    boxes = out[:, :4]
    scores = out[:, 4] * out[:, 5:].max(1)
    cls_ids = out[:, 5:].argmax(1)
    keep = scores > CONF_THRES
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    results = []
    for c in np.unique(cls_ids):
        name = CLASSES[int(c)]
        if name not in SHOW:
            continue
        idx = np.where(cls_ids == c)[0]
        b = boxes[idx]
        xyxy = np.stack([
            (b[:, 0] - b[:, 2] / 2) / r, (b[:, 1] - b[:, 3] / 2) / r,
            (b[:, 0] + b[:, 2] / 2) / r, (b[:, 1] + b[:, 3] / 2) / r], 1)
        rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                 for x1, y1, x2, y2 in xyxy]
        keep_idx = cv2.dnn.NMSBoxes(rects, scores[idx].tolist(), CONF_THRES, NMS_THRES)
        for k in np.array(keep_idx).flatten():
            x1, y1, x2, y2 = xyxy[k]
            results.append({"box": [float(x1), float(y1), float(x2), float(y2)],
                            "cls": name, "score": float(scores[idx][k])})
    return results


async def handler(ws):
    print("Cliente conectado")
    try:
        async for message in ws:
            if isinstance(message, (bytes, bytearray)):
                arr = np.frombuffer(message, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                dets = detect(img)
                await ws.send(json.dumps({"dets": dets, "w": img.shape[1], "h": img.shape[0]}))
    except websockets.exceptions.ConnectionClosed:
        pass
    print("Cliente desconectado")


async def main():
    print(f"Servidor WebSocket en 0.0.0.0:{PORT}")
    async with websockets.serve(handler, "0.0.0.0", PORT, max_size=8 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
