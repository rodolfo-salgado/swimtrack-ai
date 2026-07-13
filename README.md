# SwimTrack AI

Servicio privado de inferencia RT-DETRv2 + ByteTrack para batches de frames. La API recibe varios frames por request, pero la versión actual ejecuta TensorRT con batch interno fijo de 1 y actualiza ByteTrack secuencialmente. Esto mantiene el contrato HTTP preparado para optimizar el detector a un batch GPU dinámico sin cambiar el front.

## Arquitectura

```text
swimtrack-front (BFF)
  -> crea una tracking session
  -> POST multipart de frames ordenados
swimtrack-ai (una réplica, una GPU)
  -> RT-DETRv2(frame 0), ..., RT-DETRv2(frame N)
  -> ByteTrack(frame 0), ..., ByteTrack(frame N)
  -> bboxes en coordenadas del video original
```

El servicio mantiene estado en memoria. Debe ejecutarse con un solo worker y una sesión debe permanecer en la misma instancia durante toda su vida. El Compose ya fija `--workers 1` y reserva únicamente la GPU 0.

## Requisitos del host GPU

- Linux x86_64 con Docker Engine y Docker Compose.
- NVIDIA driver 580.173.02 o compatible.
- NVIDIA Container Toolkit configurado para Docker.
- Los submodules `vendor/ByteTrack` y `vendor/RT-DETRv2` inicializados.
- El ONNX y su archivo de external data en el directorio montado como `/model-source`.

El engine TensorRT no se distribuye. Se construye desde el ONNX la primera vez y se guarda en el volumen `trt-cache`. Un manifest incluye el hash del ONNX y su external data, versión de TensorRT, device y opciones de build; cualquier cambio invalida el cache. Si el engine compatible por manifest no puede deserializarse, el servicio lo reconstruye una vez.

## Configuración y despliegue

```bash
cp .env.example .env
```

Edita al menos `SWIMTRACK_AUTH_TOKEN` y, si es necesario, `SWIMTRACK_MODEL_HOST_DIR`. El directorio de modelos debe contener:

```text
rtdetrv2_s.onnx
rtdetrv2_s.onnx.data
```

Luego inicia el servicio:

```bash
docker compose up --build
```

El startup inicial puede tardar varios minutos mientras TensorRT construye el engine. Uvicorn no empieza a servir requests hasta que termina el lifespan de startup:

```bash
curl http://localhost:8001/healthz
curl http://localhost:8001/readyz
```

Una vez iniciado, `/healthz` indica que el proceso está vivo y `/readyz` que modelo y tracker están disponibles. Si la inicialización falla, el proceso termina para que la política `restart` de Compose vuelva a intentarlo.

El puerto escucha en `127.0.0.1` por defecto. Si el front está en otra máquina, publica `SWIMTRACK_BIND_HOST` únicamente sobre la IP de una red privada/VPN o coloca un reverse proxy con TLS/mTLS delante; no envíes el token compartido por Internet mediante HTTP plano. Aplica también límites de request en ese proxy, ya que un body HTTP chunked no siempre declara `Content-Length`.

## Contrato API

Todos los endpoints `/v1/*` requieren:

```http
X-Swimtrack-Auth: <SWIMTRACK_AUTH_TOKEN>
```

### Crear una sesión

```http
POST /v1/tracking-sessions
Content-Type: application/json

{"fps": 60}
```

Respuesta HTTP 201:

```json
{"session_id":"72528314-e373-470e-92ec-4bd3015839a7","next_sequence":0,"expires_in_seconds":900}
```

### Procesar un batch

```http
POST /v1/tracking-sessions/{session_id}/batches
Content-Type: multipart/form-data
```

El multipart contiene:

- Uno o más campos repetidos `frames`, en orden temporal. Se aceptan JPEG, PNG y WebP.
- Un campo texto `metadata` con JSON.

```json
{
  "batch_id": "video-42-0007",
  "sequence": 7,
  "frames": [
    {
      "frame_index": 56,
      "time_ms": 933.33,
      "original_width": 1920,
      "original_height": 1080
    }
  ]
}
```

Los frames transportados pueden estar redimensionados a 640×640. `original_width` y `original_height` se pasan a RT-DETRv2 como `orig_target_sizes`; las bboxes y dimensiones de la respuesta siempre pertenecen al video original.

```json
{
  "session_id": "72528314-e373-470e-92ec-4bd3015839a7",
  "batch_id": "video-42-0007",
  "sequence": 7,
  "next_sequence": 8,
  "frames": [
    {
      "frame_index": 56,
      "time_ms": 933.33,
      "width": 1920,
      "height": 1080,
      "boxes": [
        {"id":3,"x1":417.2,"y1":201.8,"x2":722.1,"y2":811.0,"conf":0.94,"class_id":0}
      ]
    }
  ]
}
```

`sequence` debe comenzar en 0 y aumentar exactamente de uno en uno. `frame_index` debe ser estrictamente creciente. ByteTrack es stateful y por eso batches de una misma sesión no pueden procesarse en paralelo.

Reenviar el mismo `batch_id`, metadata y bytes retorna la respuesta cacheada sin volver a ejecutar RT-DETRv2 ni avanzar ByteTrack. Reutilizar el `batch_id` con otro payload o enviar una secuencia fuera de orden retorna HTTP 409. La cache conserva los últimos 32 batches por defecto; un retry más antiguo que esa ventana ya no es idempotente y será rechazado por secuencia.

### Cerrar una sesión

```http
DELETE /v1/tracking-sessions/{session_id}
```

Retorna HTTP 204. Las sesiones abandonadas expiran después de `SWIMTRACK_SESSION_TTL_SECONDS`.

## Desarrollo sin GPU

El backend fake detecta regiones claras en frames y permite probar el contrato completo:

```bash
SWIMTRACK_BACKEND=fake SWIMTRACK_AUTH_TOKEN=local-test uv run uvicorn swimtrack_ai.main:app --port 8001
```

Tests y lint:

```bash
uv run pytest
uv run ruff check .
```

No uses más de un worker: el detector TensorRT y las sesiones ByteTrack viven dentro del proceso.
