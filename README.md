# vigilante-recognition

## Objetivo

`vigilante-recognition` es el subsistema responsable de detectar presencia humana, construir tracks por cámara, evaluar rostro usable, extraer embeddings, correlacionar apariciones y emitir decisiones explicables para seguridad y operación.

## Alcance del primer slice

Este primer slice **no** implementa todavía reconocimiento facial completo ni descriptor semántico.

Su objetivo es dejar un bootstrap funcional para:

- consumir `frame.ingested`
- crear o actualizar `human_track`
- consolidar presencia humana básica
- emitir:
  - `human_presence_detected`
  - `human_presence_no_face`
- persistir entidades mínimas:
  - `human_track`
  - `observed_subject`
  - `recognition_event`
  - `event_outbox`

## Fuera de alcance por ahora

- matching facial completo
- `candidate_match`
- correlación cross-camera
- descriptor semántico con Hugging Face
- integración real con media
- integración real con alerting
- revisión humana

## Arranque local

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m app.worker --fixture tests/fixtures/frame_ingested_example.json
```

## Contrato que consume

- `frame.ingested`

## Contratos que emite

- `human_presence_detected`
- `human_presence_no_face`
