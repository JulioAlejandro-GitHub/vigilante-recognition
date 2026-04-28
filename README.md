# vigilante-recognition

## Objetivo

`vigilante-recognition` es el subsistema responsable de detectar presencia humana, construir tracks por cámara, evaluar rostro usable, extraer embeddings, correlacionar apariciones y emitir decisiones explicables para seguridad y operación.

## Alcance del Slice 7

Este slice vuelve operativa la arquitectura semántica enchufable introducida en Slice 6, manteniendo intacto el flujo ya funcional de Slice 1–6.

Su objetivo actual es dejar un bootstrap funcional para:

- consumir `frame.ingested`
- crear o actualizar `human_track`
- consolidar presencia humana básica
- intentar detectar rostro en el frame usando OpenCV
- aplicar quality gate de rostro usable
- generar embedding facial local con backend liviano
- comparar contra galería conocida
- correlacionar apariciones entre cámaras usando embedding, ventana temporal y cambio de cámara
- detectar conflicto cuando identidad y continuidad técnica se contradicen
- generar descriptor semántico estructurado para sujetos no identificados o sin rostro usable
- seleccionar backend semántico configurable con backend real canónico `Qwen/Qwen2.5-VL-3B-Instruct`
- usar fallback `HuggingFaceTB/SmolVLM2-2.2B-Instruct` cuando falle el backend principal
- degradar a backend local y determinístico para tests, CI y desarrollo liviano
- elevar recurrencia de sujetos no resueltos usando similitud semántica, señal visual disponible y proximidad temporal
- sugerir revisión manual y caso técnico cuando la evidencia acumulada lo justifica
- emitir revisión manual cuando la correlación no es suficientemente confiable
- emitir:
  - `face_detected_identified`
  - `face_detected_unidentified`
  - `human_presence_no_face`
  - `cross_camera_subject_correlated`
  - `identity_conflict`
  - `manual_review_required`
  - `recurrent_unresolved_subject`
  - `case_suggestion_created`
- persistir entidades mínimas:
  - `human_track`
  - `observed_subject`
  - `recognition_event`
  - `event_outbox`
  - `cross_camera_correlation`

### Decisión del Slice 7

- si hay presencia humana confirmada, se detecta un rostro con `quality_score >= 0.75` y el match supera `FACE_MATCH_THRESHOLD` con margen suficiente, el worker emite `face_detected_identified`
- si hay presencia humana confirmada y se detecta un rostro usable pero sin match confiable, el worker emite `face_detected_unidentified`
- si no se detecta un rostro usable, el worker emite `human_presence_no_face`
- si aparece un candidato cross-camera con suficiente evidencia, el worker re-vincula el `human_track` al `observed_subject` existente y además emite `cross_camera_subject_correlated`
- si la continuidad técnica contradice una identidad conocida previa con fuerza comparable, el worker emite `identity_conflict` y `manual_review_required`
- si la correlación no alcanza umbral automático pero sí suficiente señal para no descartar, el worker emite `manual_review_required`
- la continuidad cross-camera solo se evalúa cuando la aparición abre un `human_track` nuevo; replays del mismo `correlation_id` no recalculan contra historial nuevo
- si el `human_track` ya tenía una resolución de continuidad persistida, el replay la reutiliza y reemite el mismo resultado técnico sin volver a correlacionar
- si el sujeto queda no resuelto, el worker genera un descriptor semántico estructurado y lo persiste en metadata/payload
- si un sujeto no resuelto reaparece con similitud suficiente, el worker reutiliza el `observed_subject`, eleva su recurrencia y emite `recurrent_unresolved_subject`
- si la recurrencia no resuelta acumula evidencia suficiente, el worker emite `manual_review_required`
- si la recurrencia no resuelta alcanza umbral de caso técnico, el worker emite `case_suggestion_created` con `requires_case_evaluation=true`
- la metadata facial y el resumen de matching se guardan dentro de `recognition_event.payload`
- la continuidad y el estado de resolución se guardan en metadata de `observed_subject` y `human_track`
- si `recognition.person_profile_projection` y `recognition.person_profile_embedding_projection` no tienen galería compatible disponible, se usa `app/data/dev_known_face_gallery.json` como fallback local de desarrollo
- el backend actual de embedding es `simple_face_crop_512`, preparado para reemplazarse por un motor real después
- la capa semántica se resuelve por selector de backends:
  - `qwen_vl` para `Qwen/Qwen2.5-VL-3B-Instruct`
  - `smolvlm` para `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
  - `simple` para `simple_color_signature_v1`
- si `SEMANTIC_USE_REAL_VLM=false`, los backends VLM se omiten y el worker cae al backend simple sin romper el flujo
- si `SEMANTIC_USE_REAL_VLM=true`, el backend real se ejecuta en subprocess aislado para proteger el worker ante timeouts o fallos de carga/inferencia
- el selector de device soporta `cpu`, `mps`, `cuda` y `auto`, con resolución robusta y trazabilidad en `semantic_descriptor.generation_trace`
- si el backend principal devuelve salida inválida, hace timeout o falla al cargar/inferir, el worker intenta fallback y registra la traza de intentos en `semantic_descriptor.generation_trace`
- si el fallback también falla, el worker degrada al backend simple sin descargar modelos adicionales

### Resolución de cámara en Slice 1

- `frame.ingested.payload.camera_id` debe llegar ya como UUID canónico de `api.camera.camera_id`.
- El worker valida ese valor como UUID antes de persistirlo.
- Ese UUID se persiste directamente en `human_track.camera_id`, `recognition_event.camera_id` y `observed_subject.first_camera_id` / `last_camera_id`.
- `vigilante-recognition` no depende de `recognition.camera_ref` en este slice.
- Si se necesita conservar una clave lógica externa, debe viajar en un campo separado como `payload.external_camera_key`, sin usarse como FK operativa.

## Fuera de alcance por ahora

- `candidate_match` completo de producción
- correlación cross-camera avanzada
- merge de subjects
- integración real con media
- integración real con alerting
- revisión humana completa

## Arranque local

1. Crea el entorno virtual y actívalo:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configura tu entorno:
Copia `.env.example` a `.env` y asegúrate de que los datos de conexión a Postgres sean correctos. La base de datos y esquemas (`recognition`, `outbox`) deben estar creados y asignados al usuario que configures.
El fixture y los mensajes de entrada deben traer `payload.camera_id` como UUID válido.

3. Ejecuta los tests:
```bash
PYTHONPATH=. pytest
```

4. Ejecuta exactamente estos smoke tests base:
```bash
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_no_face.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_unidentified.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_case_suggestion_created.json
```

5. Si quieres recorrer fixtures adicionales, también siguen disponibles:
```bash
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_example.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_identified.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_cross_camera_positive.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_identity_conflict.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_manual_review_required.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_recurrent_unresolved.json
```

## Integración local con vigilante-ingestion

El worker también puede consumir eventos reales `frame.ingested` desde el outbox
JSONL generado por `vigilante-ingestion`.

Flujo reproducible:

```bash
cd ../vigilante-ingestion
source .venv/bin/activate
PYTHONPATH=. python -m app.main --source-file samples/cam01.mp4 --camera-id 11111111-1111-1111-1111-111111111111 --fps 1 --max-frames 10

cd ../vigilante-recognition
source .venv/bin/activate
PYTHONPATH=. pytest
PYTHONPATH=. python -m app.worker --ingestion-jsonl ../vigilante-ingestion/outbox/frame_ingested.jsonl
PYTHONPATH=. python -m app.worker --ingestion-jsonl ../vigilante-ingestion/outbox/frame_ingested.jsonl
```

El modo JSONL reutiliza el mismo pipeline que `--fixture`, por lo que persiste en:

- `recognition.observed_subject`
- `recognition.human_track`
- `recognition.recognition_event`
- `outbox.event_outbox`

La segunda corrida del mismo archivo no debe reprocesar las líneas ya consumidas.
El worker registra contadores de `read`, `processed`, `skipped_checkpoint`,
`skipped_duplicate`, `rejected` y `frame_resolution_errors`.

Para forzar un replay completo desde el byte 0:

```bash
PYTHONPATH=. python -m app.worker --ingestion-jsonl ../vigilante-ingestion/outbox/frame_ingested.jsonl --force-replay
```

`--force-replay` ignora el checkpoint y el dedupe persistido para esa corrida,
pero sigue evitando duplicados de `event_id` dentro del mismo archivo leído.

### Checkpoint, idempotencia y rejected events

El consumo JSONL usa estado local simple bajo `.runtime/ingestion/`:

- `.runtime/ingestion/checkpoints.json`: checkpoint por path resuelto y byte offset
- `.runtime/ingestion/processed_events.json`: registro local de `event_id` procesados
- `.runtime/ingestion/rejected_events.jsonl`: DLQ local de eventos inválidos o no procesables

Se pueden cambiar por CLI:

```bash
PYTHONPATH=. python -m app.worker \
  --ingestion-jsonl ../vigilante-ingestion/outbox/frame_ingested.jsonl \
  --ingestion-checkpoint-path .runtime/ingestion/checkpoints.json \
  --ingestion-deduper-path .runtime/ingestion/processed_events.json \
  --ingestion-rejected-path .runtime/ingestion/rejected_events.jsonl
```

El worker no aborta por una línea aislada inválida. Registra en rejected events
al menos `reason`, `source_path`, `line_number`, `offset`, `event_id` si existe,
`event_type` si existe, `rejected_at` y detalles del error. Los motivos cubiertos
incluyen JSON inválido, evento sin `event_type`, `event_type` no soportado,
`camera_id` inválido, falta de `frame_ref`/`frame_uri` y frame no resoluble.

### Continuidad local entre frames

Antes de invocar el pipeline de recognition, el runner aplica una continuidad
temporal ligera para JSONL:

- agrupa por misma cámara UUID, `payload.source_type`, `payload.external_camera_key`
  y `payload.metadata.source_uri`
- reutiliza el track local reciente si el siguiente frame cae dentro de
  `INGESTION_TRACK_CONTINUITY_WINDOW_SECONDS`
- reemplaza `context.correlation_id` por una clave `local_track_*` estable para
  ese tramo y conserva el valor anterior en `context.original_correlation_id`
- agrega `context.track_continuity` con estrategia, estado y señales usadas

Esto permite que frames consecutivos de una misma fuente actualicen el mismo
`human_track` básico en vez de abrir una aparición totalmente independiente por
cada frame. No es tracking multicámara ni reemplaza la correlación avanzada.

### Resolución de frames

La estrategia de resolución es:

1. si `payload.frame_uri` existe y apunta a un archivo local, se usa ese path;
2. si `payload.frame_uri` o `payload.frame_ref` usa `s3://bucket/key` o
   `minio://bucket/key`, se descarga desde el endpoint S3/MinIO configurado;
3. si no, se intenta `payload.frame_ref`;
4. si el valor es relativo, se resuelve contra el directorio actual y luego contra
   `INGESTION_FRAME_SEARCH_ROOTS`, si está configurado.

`payload.frame_ref` sigue siendo el campo canónico del contrato. Cuando el loader
necesita usar `frame_uri` o descargar un remoto para abrir el archivo, pasa al
pipeline una copia del mensaje con `payload.frame_ref` resuelto al path físico y
conserva los valores originales en `payload.metadata.original_frame_ref` /
`payload.metadata.original_frame_uri`. `payload.frame_uri` se conserva como URI
remoto o alias práctico.

Para MinIO local:

```bash
STORAGE_S3_ENDPOINT=localhost:9000
STORAGE_S3_ACCESS_KEY=minio
STORAGE_S3_SECRET_KEY=minio123
STORAGE_S3_SECURE=false
STORAGE_S3_CACHE_DIR=.runtime/ingestion/frame-cache
```

`s3://vigilante-frames/frames/cam01/frame.jpg` se interpreta como
bucket `vigilante-frames` y object key `frames/cam01/frame.jpg`. Recognition
descarga el objeto a un cache local determinístico bajo `STORAGE_S3_CACHE_DIR` y
usa ese archivo temporal/cacheado en el pipeline actual de OpenCV/embeddings.
Los errores de bucket inexistente, objeto inexistente, credenciales inválidas,
endpoint inválido, timeout o URI mal formada se registran como
`frame_resolution_failed` en rejected events o se envían a la DLQ del broker según
el modo de consumo.

Este diseño mantiene el puente para `vigilante-media`: a futuro el resolver puede
reemplazar el acceso directo S3 por un media service sin cambiar el contrato
`frame_ref` / `frame_uri`.

### Flujo local con MinIO/S3 compartido

```bash
docker compose -f ../vigilante-docs/docker/docker-compose.support.yml up -d minio rabbitmq

cd ../vigilante-ingestion
source .venv/bin/activate
PYTHONPATH=. python -m app.main \
  --source-file samples/cam01.mp4 \
  --camera-id 11111111-1111-1111-1111-111111111111 \
  --fps 1 \
  --max-frames 10 \
  --storage-backend minio \
  --minio-endpoint localhost:9000 \
  --minio-access-key minio \
  --minio-secret-key minio123 \
  --minio-bucket vigilante-frames \
  --publish-mode both

cd ../vigilante-recognition
source .venv/bin/activate
PYTHONPATH=. python -m app.worker \
  --ingestion-jsonl ../vigilante-ingestion/outbox/frame_ingested.jsonl
```

Para RabbitMQ directo, cambia el último comando por:

```bash
PYTHONPATH=. python -m app.worker --rabbitmq-consumer --rabbitmq-max-messages 10
```

## Integración RabbitMQ con vigilante-ingestion

Slice 3 agrega consumo real AMQP sin quitar los modos previos. El worker soporta:

- `--fixture`
- `--ingestion-jsonl`
- `--rabbitmq-consumer`

Topología RabbitMQ:

- exchange principal: `vigilante.frames`
- routing key: `frame.ingested`
- cola consumida por recognition: `vigilante.recognition.frame_ingested`
- DLX: `vigilante.frames.dlx`
- DLQ: `vigilante.recognition.frame_ingested.dlq`
- routing key DLQ: `frame.ingested.dlq`

Flujo local reproducible:

```bash
docker compose -f ../vigilante-docs/docker/docker-compose.support.yml up -d rabbitmq

cd ../vigilante-ingestion
source .venv/bin/activate
PYTHONPATH=. python -m app.main \
  --source-file samples/cam01.mp4 \
  --camera-id 11111111-1111-1111-1111-111111111111 \
  --fps 1 \
  --max-frames 10 \
  --publish-mode rabbitmq

cd ../vigilante-recognition
source .venv/bin/activate
PYTHONPATH=. pytest
PYTHONPATH=. python -m app.worker \
  --rabbitmq-consumer \
  --rabbitmq-max-messages 10
```

`--rabbitmq-max-messages` es útil para smoke tests porque el proceso termina
después de consumir N deliveries. Sin ese flag el consumer queda corriendo hasta
interrupción manual.

### Ack, retry y DLQ

El consumer hace `ack` solo después de validar contrato, resolver el frame,
ejecutar el pipeline y marcar el `event_id` como procesado en el deduper local.

Van directo a DLQ del broker con `basic_reject(requeue=false)`:

- JSON inválido o no UTF-8
- evento no objeto
- `event_type` ausente o distinto de `frame.ingested`
- payload incompleto
- `payload.camera_id` no UUID
- falta de `frame_ref`/`frame_uri`
- frame físico no resoluble

Errores del pipeline se tratan como transitorios: el consumer republica el mismo
body a `vigilante.frames` con header `x-retry-count` incrementado y hace `ack`
del delivery original. Al superar `RABBITMQ_RETRY_LIMIT`, rechaza el mensaje
actual sin requeue y RabbitMQ lo enruta a la DLQ. Además se conserva el rejected
events local para trazabilidad del motivo exacto.

La idempotencia por `event_id` sigue usando
`.runtime/ingestion/processed_events.json`. Si RabbitMQ redelivera un evento ya
procesado, recognition lo salta y hace `ack`, sin volver a persistir resultados.

La continuidad básica entre frames usa el mismo `TrackContinuityService` del
modo JSONL antes de invocar el pipeline.

### Backend VLM opcional

El worker puede usar un backend VLM real, pero no es obligatorio para tests ni smoke tests.

- Para tests y CI: deja `SEMANTIC_USE_REAL_VLM=false`
- Para entorno real: configura `SEMANTIC_USE_REAL_VLM=true`
- Backend canónico: `SEMANTIC_VLM_PRIMARY_MODEL=Qwen/Qwen2.5-VL-3B-Instruct`
- Fallback: `SEMANTIC_VLM_FALLBACK_MODEL=HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Device: `SEMANTIC_DEVICE=auto|cpu|mps|cuda`
- Fallback automático: `SEMANTIC_ENABLE_FALLBACK=true`

Las dependencias VLM están aisladas para no volver pesada la instalación base:

```bash
pip install -r requirements-vlm.txt
```

Validación local explícita del backend real:

```bash
export SEMANTIC_USE_REAL_VLM=true
export SEMANTIC_DESCRIPTOR_BACKEND=qwen_vl
export SEMANTIC_ENABLE_FALLBACK=true
export SEMANTIC_DEVICE=auto
export SEMANTIC_TIMEOUT_SECONDS=45
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_no_face.json
```

Si `qwen_vl` falla por timeout, carga o inferencia, el worker intenta `smolvlm` y luego degrada a `simple`. Para desarrollo liviano y CI, mantén `SEMANTIC_USE_REAL_VLM=false`.

## Fixtures incluidos

- `tests/fixtures/frame_ingested_example.json`: caso con rostro usable y sin match confiable
- `tests/fixtures/frame_ingested_unidentified.json`: alias explícito del caso usable sin match
- `tests/fixtures/frame_ingested_no_face.json`: caso con rostro detectado pero de calidad insuficiente
- `tests/fixtures/frame_ingested_identified.json`: caso con rostro usable y match positivo
- `tests/fixtures/frame_cross_camera_positive.json`: aparición en otra cámara correlacionable con el mismo sujeto
- `tests/fixtures/frame_identity_conflict.json`: misma continuidad técnica pero identidad conocida incompatible
- `tests/fixtures/frame_manual_review_required.json`: correlación incierta que eleva revisión manual
- `tests/fixtures/frame_recurrent_unresolved.json`: sujeto sin rostro usable que reaparece con descriptor semántico consistente
- `tests/fixtures/frame_case_suggestion_created.json`: tercera aparición consistente de sujeto no resuelto que eleva sugerencia de caso
- `tests/fixtures/semantic_vlm_raw_response.json`: muestra de salida VLM para validar normalización estructurada
- `tests/fixtures/images/face_detectable.jpg`: imagen base con rostro detectable
- `tests/fixtures/images/face_low_quality.jpg`: versión degradada para forzar `human_presence_no_face`
- `tests/fixtures/images/face_identified.jpg`: rostro conocido para match positivo de desarrollo
- `tests/fixtures/images/gallery_known_biden.jpg`: segunda identidad de galería para validar margen entre candidatos
- `tests/fixtures/images/face_manual_review.jpg`: rostro usable para caso incierto sin match confiable
- `app/data/dev_known_face_gallery.json`: galería local mínima usada solo cuando no hay proyecciones activas compatibles en la base
- `app/data/dev_known_face_gallery_conflict.json`: galería local de desarrollo para forzar conflicto de identidad
- `app/data/dev_known_face_gallery_obama_only.json`: galería local de desarrollo para forzar revisión manual sin match positivo

## Configuración mínima

- `FACE_QUALITY_THRESHOLD=0.75`
- `FACE_MATCH_THRESHOLD=0.82`
- `SECOND_BEST_MARGIN=0.05`
- `EMBEDDING_BACKEND=simple_face_crop_512`
- `CROSS_CAMERA_MATCH_THRESHOLD=0.85`
- `CROSS_CAMERA_TIME_WINDOW_SECONDS=600`
- `IDENTITY_CONFLICT_MARGIN=0.25`
- `MANUAL_REVIEW_THRESHOLD=0.35`
- `SEMANTIC_DESCRIPTOR_BACKEND=qwen_vl`
- `SEMANTIC_USE_REAL_VLM=false`
- `SEMANTIC_VLM_PRIMARY_MODEL=Qwen/Qwen2.5-VL-3B-Instruct`
- `SEMANTIC_VLM_FALLBACK_MODEL=HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- `SEMANTIC_DEVICE=auto`
- `SEMANTIC_TIMEOUT_SECONDS=45`
- `SEMANTIC_ENABLE_FALLBACK=true`
- `SEMANTIC_SIMILARITY_THRESHOLD=0.72`
- `RECURRENT_SUBJECT_THRESHOLD=0.78`
- `CASE_SUGGESTION_THRESHOLD=0.9`
- `INGESTION_JSONL_PATH=../vigilante-ingestion/outbox/frame_ingested.jsonl`
- `INGESTION_FRAME_SEARCH_ROOTS=` separado por comas cuando `frame_ref` es relativo
- `INGESTION_CHECKPOINT_PATH=.runtime/ingestion/checkpoints.json`
- `INGESTION_DEDUPER_PATH=.runtime/ingestion/processed_events.json`
- `INGESTION_REJECTED_EVENTS_PATH=.runtime/ingestion/rejected_events.jsonl`
- `INGESTION_TRACK_CONTINUITY_WINDOW_SECONDS=15`
- `RABBITMQ_FRAME_EXCHANGE=vigilante.frames`
- `RABBITMQ_FRAME_ROUTING_KEY=frame.ingested`
- `RABBITMQ_FRAME_QUEUE_NAME=vigilante.recognition.frame_ingested`
- `RABBITMQ_FRAME_DLX=vigilante.frames.dlx`
- `RABBITMQ_FRAME_DLQ=vigilante.recognition.frame_ingested.dlq`
- `RABBITMQ_FRAME_DLQ_ROUTING_KEY=frame.ingested.dlq`
- `RABBITMQ_PREFETCH_COUNT=10`
- `RABBITMQ_RETRY_LIMIT=3`
- `RABBITMQ_IDLE_TIMEOUT_SECONDS=1`

## Pendiente después del Slice 7

- reemplazar el backend liviano por un motor facial real como InsightFace
- `candidate_match`
- correlación cross-camera avanzada
- optimización de rendimiento VLM real para operación sostenida de producción
- resolución MinIO / `vigilante-media`
- consumer RabbitMQ distribuido con más de una instancia y métricas formales
- integración real con alerting
- revisión humana

## Contrato que consume

- `frame.ingested`

## Contratos que emite

- `face_detected_identified`
- `face_detected_unidentified`
- `human_presence_no_face`
- `cross_camera_subject_correlated`
- `identity_conflict`
- `manual_review_required`
- `recurrent_unresolved_subject`
- `case_suggestion_created`
