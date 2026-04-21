# Model Selection

## Slice 1
No usar todavía InsightFace ni VLM en la ejecución real del primer slice.

## Slice 2
- rostro: InsightFace + ONNX Runtime
- descriptor semántico: `Qwen/Qwen2.5-VL-3B-Instruct`
- fallback: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
