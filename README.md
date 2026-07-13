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

El servicio mantiene estado en memoria. Debe ejecutarse con un solo worker y una sesión debe permanecer en la misma instancia durante toda su vida. La configuración nativa usa `CUDA_VISIBLE_DEVICES=0` y `--workers 1`; el Compose aplica las restricciones equivalentes.

## Requisitos del host GPU

- Linux x86_64 con glibc 2.28 o posterior, NVIDIA driver R580 y acceso funcional a la GPU mediante `nvidia-smi`.
- `uv` con capacidad de instalar Python 3.12 y descargar paquetes desde PyPI.
- Los submodules `vendor/ByteTrack` y `vendor/RT-DETRv2` inicializados.
- El ONNX `rtdetrv2_s.onnx` y su archivo `rtdetrv2_s.onnx.data` disponibles en el filesystem.
- Al menos 8 GiB libres y escribibles para `.venv`, los wheels CUDA/TensorRT y el engine cacheado; la instalación comprobada ocupa aproximadamente 3.5 GiB antes del cache y algunas configuraciones pueden conservar otra copia de los wheels.

El engine TensorRT no se distribuye. Se construye desde el ONNX la primera vez y se guarda en el directorio configurado por `SWIMTRACK_MODEL_CACHE_DIR`. Un manifest incluye el hash del ONNX y su external data, versión de TensorRT, device y opciones de build; cualquier cambio invalida el cache. Si el engine compatible por manifest no puede deserializarse, el servicio lo reconstruye una vez.

## Ejecución nativa con `uv` sin Docker ni sudo

Esta es la ruta recomendada cuando el usuario no puede instalar paquetes del sistema. El extra `native-gpu` instala TensorRT 10.13.3.9 y CUDA Runtime 13.0 dentro de `.venv`; no instala ni modifica el driver del host.

Desde `swimtrack-ai/`:

```bash
nvidia-smi
ldd --version
git submodule update --init --recursive
cp .env.native.example .env.native
mkdir -p model-cache
uv python install 3.12
uv sync --locked --no-dev --extra native-gpu
```

Las rutas de `.env.native.example` son relativas a `swimtrack-ai/` y funcionan con la estructura de directorios recomendada. Edita `SWIMTRACK_AUTH_TOKEN` y usa rutas absolutas si los repositorios no son hermanos.

Comprueba TensorRT y la visibilidad CUDA antes de iniciar la API:

```bash
uv run --locked --no-dev --extra native-gpu --env-file .env.native -- python -c "from cuda.bindings import runtime as c; error, count = c.cudaGetDeviceCount(); print(error, count); assert error == c.cudaError_t.cudaSuccess and count > 0"
uv run --locked --no-dev --extra native-gpu --env-file .env.native -- python -c "import tensorrt as trt; print(trt.__version__); assert trt.Builder(trt.Logger())"
```

Inicia exactamente un worker y mantén el servicio ligado a localhost cuando se acceda por SSH tunnel:

```bash
uv run --locked --no-dev --extra native-gpu --env-file .env.native -- uvicorn swimtrack_ai.main:app --host 127.0.0.1 --port 8001 --workers 1
```

El startup inicial puede tardar varios minutos mientras TensorRT construye el engine. Uvicorn no acepta requests hasta terminar el lifespan de startup. Cuando aparezca `Application startup complete`, verifica:

```bash
curl http://127.0.0.1:8001/healthz
curl http://127.0.0.1:8001/readyz
```

Para una primera prueba persistente sin privilegios, ejecuta Uvicorn dentro de `tmux`. En un cluster administrado usa el scheduler disponible, por ejemplo Slurm, en vez de dejar procesos fuera de una asignación.

### Conexión mediante SSH tunnel

Desde la máquina donde corre `swimtrack-front`, crea un tunnel cifrado y deja el proceso activo:

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18001:127.0.0.1:8001 usuario@gpu-host
```

Configura el front con `VISION_BASE_URL=http://127.0.0.1:18001` y el mismo token. Esta alternativa no requiere abrir el puerto 8001 ni administrar el firewall de la máquina GPU.

## Ejecución alternativa con Docker

El despliegue containerizado se conserva para hosts donde Docker y NVIDIA Container Toolkit estén disponibles:

```bash
cp .env.example .env
docker compose up --build
```

Edita al menos `SWIMTRACK_AUTH_TOKEN` y `SWIMTRACK_MODEL_HOST_DIR`. El volumen `trt-cache` persiste el engine y el Compose limita el proceso a GPU 0. Si el front está en otra máquina, publica `SWIMTRACK_BIND_HOST` únicamente sobre una IP privada/VPN o coloca un reverse proxy con TLS/mTLS delante.

Una vez iniciado, `/healthz` indica que el proceso está vivo y `/readyz` que modelo y tracker están disponibles. Si la inicialización falla, el proceso termina para que el supervisor elegido vuelva a intentarlo.

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
