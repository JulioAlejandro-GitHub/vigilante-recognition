# vigilante-recognition

## Objetivo

`vigilante-recognition` es el subsistema responsable de detectar presencia humana, construir tracks por cámara, evaluar rostro usable, extraer embeddings, correlacionar apariciones y emitir decisiones explicables para seguridad y operación.

## Alcance del Slice 2

Este slice **no** implementa todavía reconocimiento facial completo ni descriptor semántico.

Su objetivo actual es dejar un bootstrap funcional para:

- consumir `frame.ingested`
- crear o actualizar `human_track`
- consolidar presencia humana básica
- intentar detectar rostro en el frame usando OpenCV
- aplicar quality gate de rostro usable
- emitir:
  - `face_detected_unidentified`
  - `human_presence_no_face`
- persistir entidades mínimas:
  - `human_track`
  - `observed_subject`
  - `recognition_event`
  - `event_outbox`

### Decisión del Slice 2

- si hay presencia humana confirmada y se detecta un rostro con `quality_score >= 0.75`, el worker emite `face_detected_unidentified`
- si no se detecta un rostro usable, el worker emite `human_presence_no_face`
- la metadata facial mínima se guarda dentro de `recognition_event.payload.face_detection`
- no se calculan embeddings, no hay matching facial y no se usa InsightFace en este slice

### Resolución de cámara en Slice 1

- `frame.ingested.payload.camera_id` debe llegar ya como UUID canónico de `api.camera.camera_id`.
- El worker valida ese valor como UUID antes de persistirlo.
- Ese UUID se persiste directamente en `human_track.camera_id`, `recognition_event.camera_id` y `observed_subject.first_camera_id` / `last_camera_id`.
- `vigilante-recognition` no depende de `recognition.camera_ref` en este slice.
- Si se necesita conservar una clave lógica externa, debe viajar en un campo separado como `payload.external_camera_key`, sin usarse como FK operativa.

## Fuera de alcance por ahora

- matching facial completo
- `candidate_match`
- correlación cross-camera
- descriptor semántico con Hugging Face
- integración real con media
- integración real con alerting
- revisión humana

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

4. Ejecuta el worker usando el fixture de ejemplo:
```bash
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_example.json
```

## Fixtures incluidos

- `tests/fixtures/frame_ingested_example.json`: caso con rostro detectable y usable
- `tests/fixtures/frame_ingested_no_face.json`: caso con rostro detectado pero de calidad insuficiente
- `tests/fixtures/images/face_detectable.jpg`: imagen base con rostro detectable
- `tests/fixtures/images/face_low_quality.jpg`: versión degradada para forzar `human_presence_no_face`

## Pendiente para Slice 3

- matching facial completo
- embeddings faciales
- InsightFace
- `candidate_match`
- correlación cross-camera
- descriptor semántico con Hugging Face
- integración real con media
- integración real con alerting
- revisión humana
- Integración real con RabbitMQ en `consumer.py` y `publisher.py`

## Contrato que consume

- `frame.ingested`

## Contratos que emite

- `face_detected_unidentified`
- `human_presence_no_face`
