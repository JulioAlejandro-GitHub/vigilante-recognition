# Recognition Pipeline

## Slice 1

1. consumir `frame.ingested`
2. abrir `observed_subject`
3. abrir `human_track`
4. acumular presencia básica
5. decidir:
   - `human_presence_detected`, o
   - `human_presence_no_face`
6. persistir `recognition_event`
7. escribir outbox

## Slice 2+

- face detection
- quality gate
- embeddings
- matching
- cross-camera
- semantic descriptor
