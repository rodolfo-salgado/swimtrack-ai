# Benchmark de los videos de referencia

`scripts/benchmark_test_videos.py` ejecuta o planifica un benchmark reproducible para `test01` a `test08`. Rechaza explícitamente `test09`, ya que ese video contiene dos nadadores y no pertenece al escenario actual de un nadador por carril.

El runner guarda el SHA-256 de cada archivo de entrada, geometría, FPS, stride de sampling, configuración solicitada por el operador, respuesta de `readyz`, configuración efectiva de tracking devuelta al crear cada sesión, NDJSON por frame, métricas por video y un agregado. El token nunca se escribe en los artefactos.

## Modos

`diagnose` es el modo predeterminado y no realiza requests HTTP, TensorRT, CUDA ni NVDEC. Lee los ocho MP4 con OpenCV, calcula el plan de sampling y crea `run.json`; sirve para validar el dataset y preparar comparaciones en una máquina sin GPU.

`remote` crea una sesión independiente por video contra SwimTrack AI. `--transport video` sube el MP4 original al endpoint `/video`, exige `Content-Type: application/x-ndjson` y consume el stream ordenado, por lo que el decode ocurre en el servidor mediante NVDEC. `--transport frames` reproduce el transporte histórico del Front: OpenCV selecciona los mismos índices de fuente, aplica `INTER_LINEAR`, codifica JPEG y envía batches ordenados. El modo `frames` mide preparación local y telemetry HTTP, pero no superpone preparación y requests como el Front; para comparar inferencia conviene mirar los headers `X-Swimtrack-*-Ms` además del wall time del cliente.

## Diagnóstico local sin GPU

Desde `swimtrack-ai/`:

```bash
uv run --script scripts/benchmark_test_videos.py --mode diagnose --run-id rtdetrv2-s-video-plan --transport video --model-label rtdetrv2-s --model-artifact rtdetrv2_s.onnx --crop-label far-crop-v1
```

El resultado queda por defecto en `../results/benchmarks/<run-id>/run.json`. Añade `--no-source-hash` únicamente para un diagnóstico local rápido; los benchmarks remotos deben conservar los hashes de entrada.

## Ejecución remota en GPU 0

El despliegue administrado ya fija `CUDA_VISIBLE_DEVICES=0` y `SWIMTRACK_DEVICE=cuda:0` en `swimtrack-ansible/roles/swimtrack_ai/templates/swimtrack-ai.env.j2`. Antes de un benchmark real, publica y verifica la revisión mediante el flujo de Ansible del proyecto; el runner no cambia modelos, engines, crop ni variables del servidor.

Exporta el token sólo en la sesión de shell mediante el mecanismo seguro local de Ansible y nunca lo añadas a un comando versionado. Luego ejecuta el runner desde el controller:

```bash
export SWIMTRACK_BENCHMARK_AUTH_TOKEN='token-obtenido-de-un-secreto-local'
uv run --script scripts/benchmark_test_videos.py --mode remote --run-id rtdetrv2-s-640-video-crop --base-url http://10.0.218.101:7001 --transport video --diagnostics boxes --max-fps 30 --model-label rtdetrv2-s --model-artifact rtdetrv2_s.onnx --crop-label far-crop-v1 --metadata engine_filename=rtdetrv2_s_fp16.engine
```

Para comparar la pérdida de detalle causada por el transporte JPEG actual, cambia solamente el transporte y deja iguales el run label, el modelo desplegado, el crop, el sampling y diagnostics:

```bash
uv run --script scripts/benchmark_test_videos.py --mode remote --run-id rtdetrv2-s-640-frames-crop --base-url http://10.0.218.101:7001 --transport frames --diagnostics boxes --max-fps 30 --inference-size 640 --jpeg-quality 85 --batch-size 4 --model-label rtdetrv2-s --model-artifact rtdetrv2_s.onnx --crop-label far-crop-v1
```

Un benchmark de modelo mayor declara el artifact realmente desplegado y no intenta modificar TensorRT desde el cliente:

```bash
uv run --script scripts/benchmark_test_videos.py --mode remote --run-id rtdetrv2-m-800-video-crop --base-url http://10.0.218.101:7001 --transport video --diagnostics boxes --max-fps 30 --model-label rtdetrv2-m --model-artifact rtdetrv2_m_800.onnx --crop-label far-crop-v1 --metadata engine_filename=rtdetrv2_m_800_fp16.engine --metadata input_size=800
```

## Artefactos y lectura de métricas

Cada video produce `input.json`, `frames.ndjson` y `result.json`. `result.json.analysis.diagnostics.stages` separa `person_candidates`, `detector_accepted`, `weak_candidates`, `after_roi`, `weak_candidates_after_roi` y `active_tracks`; `funnel` muestra las retenciones de ambas rutas. `tracking.weak_reactivations` separa eventos de reacquisición débil de los gaps del mismo ID. `execution` conserva wall time, primer resultado desde el request y de extremo a extremo para video, la comparación entre muestras esperadas y emitidas, y para batches JPEG la telemetría del servidor. Si falla `readyz`, el runner conserva `run.json` con estado `failed` antes de salir.

`aggregate.json` combina conteos y cobertura sólo si todos los videos tienen diagnostics. Las métricas de coverage no son detection recall: mezclan ausencia visual real, detector, ROI y asociación de ByteTrack. No hay ground truth de bounding boxes en estos videos, por lo que el runner no presenta precision/recall de detección como si estuvieran anotados.

La configuración devuelta en `result.json.session.tracking_configuration` es la evidencia autoritativa de thresholds, ROI, crop y buffers aplicados por el servicio. `model_label`, `model_artifact`, `crop_label` y `metadata` son declaraciones del operador para vincular el resultado al deployment que se evaluó.

## Validación local

```bash
uv run pytest tests/test_benchmark_test_videos.py
uv run ruff check scripts/benchmark_test_videos.py tests/test_benchmark_test_videos.py
uv run --script scripts/benchmark_test_videos.py --mode diagnose --run-id benchmark-diagnostic-smoke --results-root /tmp/swimtrack-benchmark-smoke --no-source-hash
```
