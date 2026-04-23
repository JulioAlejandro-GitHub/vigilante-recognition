# vigilante-recognition

## Objetivo

`vigilante-recognition` es el subsistema responsable de detectar presencia humana, construir tracks por cámara, evaluar rostro usable, extraer embeddings, correlacionar apariciones y emitir decisiones explicables para seguridad y operación.

## Alcance del Slice 4

Este slice implementa continuidad mínima de `observed_subject`, correlación cross-camera, detección de conflicto de identidad y emisión de revisión manual, manteniendo intacto el flujo ya funcional de Slice 1–3.

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
- emitir revisión manual cuando la correlación no es suficientemente confiable
- emitir:
  - `face_detected_identified`
  - `face_detected_unidentified`
  - `human_presence_no_face`
  - `cross_camera_subject_correlated`
  - `identity_conflict`
  - `manual_review_required`
- persistir entidades mínimas:
  - `human_track`
  - `observed_subject`
  - `recognition_event`
  - `event_outbox`
  - `cross_camera_correlation`

### Decisión del Slice 4

- si hay presencia humana confirmada, se detecta un rostro con `quality_score >= 0.75` y el match supera `FACE_MATCH_THRESHOLD` con margen suficiente, el worker emite `face_detected_identified`
- si hay presencia humana confirmada y se detecta un rostro usable pero sin match confiable, el worker emite `face_detected_unidentified`
- si no se detecta un rostro usable, el worker emite `human_presence_no_face`
- si aparece un candidato cross-camera con suficiente evidencia, el worker re-vincula el `human_track` al `observed_subject` existente y además emite `cross_camera_subject_correlated`
- si la continuidad técnica contradice una identidad conocida previa con fuerza comparable, el worker emite `identity_conflict` y `manual_review_required`
- si la correlación no alcanza umbral automático pero sí suficiente señal para no descartar, el worker emite `manual_review_required`
- la metadata facial y el resumen de matching se guardan dentro de `recognition_event.payload`
- la continuidad y el estado de resolución se guardan en metadata de `observed_subject` y `human_track`
- si `recognition.person_profile_projection` y `recognition.person_profile_embedding_projection` no tienen galería compatible disponible, se usa `app/data/dev_known_face_gallery.json` como fallback local de desarrollo
- el backend actual de embedding es `simple_face_crop_512`, preparado para reemplazarse por un motor real después

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
- descriptor semántico con Hugging Face
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

4. Ejecuta el worker usando los fixtures principales:
```bash
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_example.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_no_face.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_ingested_identified.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_cross_camera_positive.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_identity_conflict.json
PYTHONPATH=. python3 -m app.worker --fixture tests/fixtures/frame_manual_review_required.json
```

## Fixtures incluidos

- `tests/fixtures/frame_ingested_example.json`: caso con rostro usable y sin match confiable
- `tests/fixtures/frame_ingested_unidentified.json`: alias explícito del caso usable sin match
- `tests/fixtures/frame_ingested_no_face.json`: caso con rostro detectado pero de calidad insuficiente
- `tests/fixtures/frame_ingested_identified.json`: caso con rostro usable y match positivo
- `tests/fixtures/frame_cross_camera_positive.json`: aparición en otra cámara correlacionable con el mismo sujeto
- `tests/fixtures/frame_identity_conflict.json`: misma continuidad técnica pero identidad conocida incompatible
- `tests/fixtures/frame_manual_review_required.json`: correlación incierta que eleva revisión manual
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

## Pendiente para Slice 5

- reemplazar el backend liviano por un motor facial real como InsightFace
- `candidate_match`
- correlación cross-camera avanzada
- descriptor semántico con Hugging Face
- integración real con media
- integración real con alerting
- revisión humana
- Integración real con RabbitMQ en `consumer.py` y `publisher.py`

## Contrato que consume

- `frame.ingested`

## Contratos que emite

- `face_detected_identified`
- `face_detected_unidentified`
- `human_presence_no_face`
- `cross_camera_subject_correlated`
- `identity_conflict`
- `manual_review_required`
