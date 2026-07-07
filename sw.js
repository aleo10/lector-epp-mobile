// Service worker: cachea la app y el modelo para funcionar offline,
// pero prioriza traer el codigo actualizado cuando hay conexion.
const CACHE = "epp-v3";
const ASSETS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./models/yolox_ppe.onnx",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;

  // Nunca interceptar la API/streams del relay de la camara IP (video en vivo,
  // detecciones, estado): son datos vivos, muchas veces cross-origin (Tailscale).
  const url = e.request.url;
  if (url.includes("/video_feed") || url.includes("/detections") ||
      url.includes("/status") || url.includes("/ip/")) return;

  // El modelo (grande, no cambia): cache-first para no re-descargar 35 MB.
  if (url.includes(".onnx")) {
    e.respondWith(
      caches.match(e.request).then((hit) =>
        hit || fetch(e.request).then((res) => {
          const clone = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, clone));
          return res;
        })
      )
    );
    return;
  }

  // Codigo/assets: network-first => siempre lo ultimo si hay red,
  // con fallback al cache si no hay conexion (offline).
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        if (res && res.status === 200) {
          const clone = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, clone));
        }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
