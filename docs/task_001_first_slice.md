# TASK 001 - First Slice

## Objetivo

Implementar el bootstrap funcional de `vigilante-recognition` para consumir `frame.ingested` y producir:

- `human_presence_detected`
- `human_presence_no_face`

## Debe hacer

- cargar un mensaje `frame.ingested`
- crear `observed_subject`
- crear `human_track`
- consolidar presencia humana básica
- persistir `recognition_event`
- persistir `event_outbox`
- tener tests mínimos

## No debe hacer todavía

- matching facial
- candidate_match
- correlación cross-camera
- VLM
- alerting real
- media real

## Definition of Done

- `pytest` pasa
- `python -m app.worker --fixture tests/fixtures/frame_ingested_example.json` funciona
- se genera un evento de salida coherente
