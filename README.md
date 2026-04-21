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

1. Crea el entorno virtual y actívalo:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configura tu entorno:
Copia `.env.example` a `.env` y asegúrate de que los datos de conexión a Postgres sean correctos. La base de datos y esquemas (`recognition`, `outbox`) deben estar creados y asignados al usuario que configures.

3. Ejecuta los tests:
```bash
PYTHONPATH=. pytest
```

4. Ejecuta el worker usando el fixture de ejemplo:
```bash
PYTHONPATH=. python -m app.worker --fixture tests/fixtures/frame_ingested_example.json
```

## Pendiente para Slice 2

- matching facial completo
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

- `human_presence_detected`
- `human_presence_no_face`
