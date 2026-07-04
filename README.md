# 📱 Detector de EPP — Móvil (PWA)

Versión móvil del [detector de EPP](https://github.com/aleo10/lector-epp): una **PWA** que usa el
**celular como cámara y pantalla a la vez**, para hacer demostraciones caminando. Enfocás a las
personas y la app detecta en vivo si llevan sus Elementos de Protección Personal (casco, chaleco, barbijo).

**En vivo:** https://aleo10.github.io/lector-epp-mobile/

Detecta con un modelo **YOLOX propio** (Apache 2.0, sin costo de licencia). Muestra solo las clases
relevantes de EPP: `Person`, `Hardhat`/`NO-Hardhat`, `Safety Vest`/`NO-Safety Vest`, `Mask`/`NO-Mask`.

## Doble modo de inferencia (seleccionable en la app)

Porque la conexión en el lugar de la demo puede ser incierta, la app permite elegir dónde se procesa:

| Modo | Cómo funciona | Cuándo usarlo |
|---|---|---|
| 📱 **Celular** | La cámara y el modelo corren 100% en el celular (ONNX Runtime Web + **WebGPU**). Sin conexión, sin latencia. | Conexión mala / lugar chico / gente cerca. En un Galaxy S23 Ultra da ~4.6 FPS. |
| ☁️ **RunPod** | El celular manda los frames por **WebSocket** a un servidor con GPU que hace la inferencia y devuelve las detecciones. | Buena conexión / lugar grande / gente lejana. Modelo más grande y preciso. |

El modo Celular tiene un techo de hardware: para ir fluido usa modelo liviano + baja resolución,
lo que le cuesta detectar personas lejanas. El modo RunPod levanta ese techo a cambio de depender
de la red.

## Estructura

```
index.html            La PWA completa (cámara, inferencia local, cliente WebSocket, dibujo)
models/yolox_ppe.onnx El modelo YOLOX exportado a ONNX
server/               Backend de inferencia para el modo RunPod
  infer_server.py       Servidor WebSocket + ONNX Runtime GPU
  requirements.txt
.nojekyll             Evita el procesamiento Jekyll de GitHub Pages (destraba el deploy)
```

## Modo RunPod — desplegar el backend

El modo RunPod necesita un servidor con GPU corriendo `server/infer_server.py`. Para demos conviene
un **pod on-demand** (no serverless): se enciende al empezar la demo y se apaga al terminar, así se
paga solo por las horas de uso, sin cold starts y con el WebSocket siempre conectado.

1. Crear un pod GPU en RunPod **exponiendo el puerto 8765**.
2. Subir `server/infer_server.py` y `models/yolox_ppe.onnx` al pod.
3. Instalar y arrancar:
   ```bash
   pip install -r requirements.txt
   python infer_server.py
   ```
4. RunPod expone el puerto vía su proxy con TLS automático:
   `wss://<POD_ID>-8765.proxy.runpod.net`
5. En la app, tocar **☁️ RunPod** y pegar esa URL (queda guardada en el celular).

## Notas técnicas

- **WebGPU:** hay que cargar el bundle `ort.webgpu.min.js` (el `ort.min.js` genérico cae a WASM, ~8x más lento).
- **Deploy en GitHub Pages:** el primer build a veces se cuelga; `.nojekyll` + forzar rebuild lo resuelve.
  Para saltear el caché del celular al probar cambios, agregar `?v=N` a la URL.
- **HTTPS obligatorio:** la cámara del navegador (`getUserMedia`) solo funciona sobre HTTPS — por eso
  GitHub Pages, que lo da gratis.
- **Multiplataforma:** el mismo código funciona en Android y iPhone (Safari). En iPhone, WebGPU
  requiere iOS 18+; en versiones previas usa WASM (más lento).
