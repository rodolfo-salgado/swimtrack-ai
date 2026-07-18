# SwimTrack AI

Servicio privado de inferencia RT-DETRv2 + ByteTrack para batches de frames. La API decodifica JPEG en paralelo con un límite global, agrupa las vistas full/crop en TensorRT y actualiza ByteTrack secuencialmente. El engine usa un profile B1/óptimo B4/máximo B8 cuando el ONNX declara dinámica la dimensión 0 de `images`, `orig_target_sizes`, `labels`, `boxes` y `scores`; con un artefacto antiguo de outputs B1 el servicio se limita de forma segura a B1.

## Arquitectura

```text
swimtrack-front (BFF)
  -> crea una tracking session
  -> POST multipart de frames ordenados
swimtrack-ai (una réplica, una GPU)
  -> decode JPEG limitado + RT-DETRv2(batch de vistas full/crop)
  -> ByteTrack(frame 0), ..., ByteTrack(frame N)
  -> bboxes en coordenadas del video original
```

El servicio mantiene estado en memoria. Debe ejecutarse con un solo worker y una sesión debe permanecer en la misma instancia durante toda su vida. La configuración nativa usa `CUDA_VISIBLE_DEVICES=0` y `--workers 1`; el Compose aplica las restricciones equivalentes.

## Requisitos del host GPU

- Linux x86_64 con glibc 2.28 o posterior, NVIDIA driver R580 y acceso funcional a la GPU mediante `nvidia-smi`.
- FFmpeg y FFprobe instalados en el host; `ffmpeg -hwaccels` debe incluir `cuda` para la ruta de video comprimido.
- `uv` con capacidad de instalar Python 3.12 y descargar paquetes desde PyPI.
- Los submodules `vendor/ByteTrack` y `vendor/RT-DETRv2` inicializados.
- El ONNX dinámico `artifacts/models/rtdetrv2_s.onnx` disponible en el filesystem.
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

Las rutas de `.env.native.example` son relativas a `swimtrack-ai/`. Edita `SWIMTRACK_AUTH_TOKEN` y usa rutas absolutas si almacenas el artifact fuera del repositorio.

### Artifact del modelo

El modelo no depende de otro repositorio. Genera el ONNX con batch dinámico desde el checkout actual y valida el checksum del checkpoint:

```bash
uv run --script scripts/export_rtdetrv2_onnx.py
```

El script usa `vendor/RT-DETRv2`, descarga el checkpoint oficial sólo si todavía no existe en `artifacts/checkpoints/`, y escribe `artifacts/models/rtdetrv2_s.onnx`. El ONNX usa data embebida; un `.onnx.data` sólo se acepta por compatibilidad con artifacts externos heredados.

Comprueba TensorRT y la visibilidad CUDA antes de iniciar la API:

```bash
uv run --locked --no-dev --extra native-gpu --env-file .env.native -- python -c "from cuda.bindings import runtime as c; error, count = c.cudaGetDeviceCount(); print(error, count); assert error == c.cudaError_t.cudaSuccess and count > 0"
uv run --locked --no-dev --extra native-gpu --env-file .env.native -- python -c "import tensorrt as trt; print(trt.__version__); assert trt.Builder(trt.Logger())"
```

Para pruebas locales, inicia exactamente un worker ligado a localhost:

```bash
uv run --locked --no-dev --extra native-gpu --env-file .env.native -- uvicorn swimtrack_ai.main:app --host 127.0.0.1 --port 8001 --workers 1
```

El startup inicial puede tardar varios minutos mientras TensorRT construye el engine. Uvicorn no acepta requests hasta terminar el lifespan de startup. Cuando aparezca `Application startup complete`, verifica:

```bash
curl http://127.0.0.1:8001/healthz
curl http://127.0.0.1:8001/readyz
```

Para una primera prueba persistente sin privilegios, ejecuta Uvicorn dentro de `tmux`. En un cluster administrado usa el scheduler disponible, por ejemplo Slurm, en vez de dejar procesos fuera de una asignación.

### Conexión privada directa

En el despliegue de este proyecto, Uvicorn se liga exclusivamente a la IP privada de la GPU y a un puerto permitido:

```bash
uv run --locked --no-dev --extra native-gpu --env-file .env.native -- uvicorn swimtrack_ai.main:app --host 10.0.218.101 --port 7001 --workers 1
```

En esta VM temporal, el rango `7000-7099` ya está abierto y `swimtrack-ansible` no modifica UFW. Configura el Front con `VISION_BASE_URL=http://10.0.218.101:7001` y el mismo token. Todas las rutas `/v1/*` exigen el token, pero el acceso de red no se limita por origen; no uses `0.0.0.0` ni reutilices este HTTP directo fuera de la red privada temporal del proyecto.

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

{"fps": 60, "lap_calibration_id": "fixed-camera-v1", "diagnostics": "counts"}
```

`lap_calibration_id` es opcional. `fixed-camera-v1` habilita el score heurístico de vuelta para el carril central de la cámara fija del proyecto; si se omite, la respuesta conserva el contrato anterior sin `lap_scores`.

`diagnostics` también es opcional y acepta `none`, `counts` o `boxes`. Los dos últimos añaden instrumentación por frame para `person_candidates`, detecciones aceptadas por score/área, detecciones que sobreviven el ROI de carril, tracks activos y tracks conservados como `lost`; `boxes` debe reservarse para experimentos porque aumenta el payload.

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
      ],
      "lap_scores": [
        {
          "lane_id": "center",
          "track_id": 3,
          "lap_score": 0.82,
          "no_lap_score": 0.18,
          "observation_quality": 0.93,
          "evaluable": true,
          "longitudinal_position": 0.91,
          "endpoint": "near",
          "candidate_time_ms": 810.0,
          "candidate_episode_id": 1,
          "window_start_ms": 0.0,
          "window_end_ms": 933.33,
          "score_version": "trajectory-v4",
          "evidence": {"wall":0.96,"approach":0.84,"reversal":0.88,"departure":0.79,"track_quality":0.93}
        }
      ]
    }
  ]
}
```

`lap_score` y `no_lap_score` son scores heurísticos continuos, no probabilidades calibradas ni un conteo definitivo. El score combina cercanía a la pared, aproximación, reversión, salida y calidad local de tracking. `trajectory-v4` sólo habilita candidatos después de observar al nadador dentro de la zona interior del carril; así una salida que comienza junto a la pared no se confunde con una vuelta. Cuando ByteTrack no mantiene un track, el análisis usa las detecciones de la ROI del carril y puede unir ambos lados de una oclusión de hasta 6 segundos, reduciendo la confianza según la duración del gap. `candidate_episode_id` identifica una visita completa a la pared y agrupa todos sus candidatos, mientras que `candidate_time_ms` conserva el instante estimado de contacto. Si falta suficiente trayectoria, `evaluable` es `false` y `no_lap_score` se omite para no convertir ausencia de observación en evidencia de `no_lap`.

Las respuestas `200` incluyen los headers `X-Swimtrack-Decode-Ms`, `X-Swimtrack-Process-Ms` y `X-Swimtrack-Total-Ms`. Miden, respectivamente, la lectura/decodificación multipart, el procesamiento serializado de la sesión (inference y tracking) y el total de la ruta; el Front los registra por batch sin alterar el JSON del contrato.

### Procesar un video comprimido con NVDEC

```http
POST /v1/tracking-sessions/{session_id}/video
Content-Type: multipart/form-data
```

El multipart contiene el campo `video` con el archivo original y `sample_fps` con la frecuencia de muestreo deseada. La respuesta es `application/x-ndjson`: cada línea es un objeto `FrameResult`, en orden temporal, con el `time_ms` de presentación real del video y las dimensiones originales. El Front debe consumir cada línea conforme llega; no hay un wrapper `BatchResult` en esta ruta.

FFprobe valida el stream antes de iniciar la respuesta y FFmpeg se ejecuta con `-hwaccel cuda -hwaccel_device 0 -hwaccel_output_format cuda`. El filtro selecciona usando presentation timestamps antes de `hwdownload`, por lo que sólo los frames muestreados cruzan de la GPU al proceso de inference. No existe fallback silencioso a CPU: si CUDA/NVDEC no puede decodificar, la API retorna `503` con `error.code="nvdec_decode_failed"` antes de empezar el stream cuando es posible. Una falla después de emitir líneas termina el stream y queda registrada en el servidor.

Las respuestas exitosas incluyen `X-Swimtrack-Decode-Path: nvdec` y `X-Swimtrack-Decode-Backend: ffmpeg`; esos headers sólo se envían después de decodificar el primer batch mediante la ruta CUDA. `SWIMTRACK_MAX_VIDEO_BYTES`, `SWIMTRACK_VIDEO_DECODE_BATCH_FRAMES`, `SWIMTRACK_FFMPEG_PATH`, `SWIMTRACK_FFPROBE_PATH` y `SWIMTRACK_VIDEO_PROBE_TIMEOUT_SECONDS` controlan límites y herramientas de esta ruta.

### Calibración fija de carril

La calibración `fixed-camera-v1` proviene de `mpv-shot0001.jpg` (1041×1041) y usa coordenadas normalizadas, por lo que también aplica a los videos originales 1080×1080 mientras no cambien el crop ni la cámara. Sólo el carril central es visible de pared a pared.

Con `SWIMTRACK_LANE_ROI_ENABLED=true`, las detecciones se asignan al polígono antes de ByteTrack y cada carril usa una instancia independiente del tracker. Las sesiones sin calibración mantienen el tracker global anterior.

`SWIMTRACK_FAR_CROP_ENABLED=true` agrega una segunda inference únicamente para sesiones con calibración `fixed-camera-v1`. El crop normalizado configurable corresponde por default a `(320,120)–(760,560)` en 1080p, se redimensiona al mismo input TensorRT que el frame completo, remapea sus boxes a coordenadas originales y fusiona ambos resultados con NMS antes de la ROI y ByteTrack. El full-frame permanece activo para cubrir el resto del carril.

La baseline seleccionada por el sweep de los videos 1 (`no_lap`) y 6 (`lap`) usa `SWIMTRACK_SCORE_THRESHOLD=0.15`, `SWIMTRACK_MIN_BOX_AREA=250`, `SWIMTRACK_TRACK_THRESHOLD=0.45`, `SWIMTRACK_TRACK_BUFFER=60`, `SWIMTRACK_MATCH_THRESHOLD=0.80` y `SWIMTRACK_LANE_ROI_ENABLED=true`. Estos son también los defaults de `Settings`, Compose y `.env.example`; siguen siendo configurables por environment para repetir experimentos.

```text
visible_polygon = [(0.4463,0.1583), (0.5815,0.1583), (1.0000,0.6630), (1.0000,0.9769), (0.0000,0.9769), (0.0000,0.6824)]
source_quad     = [(0.4463,0.1583), (0.5815,0.1583), (1.2603,0.9769), (-0.2507,0.9769)]
```

`source_quad` extrapola las corcheras hasta la pared cercana, fuera del encuadre, y se transforma mediante homografía a una coordenada longitudinal `s ∈ [0,1]`. Los carriles laterales visibles sólo parcialmente quedan excluidos del score.

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
