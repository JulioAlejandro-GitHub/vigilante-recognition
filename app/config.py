from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "vigilante_recognition"
    db_user: str = "julio"
    db_password: str = ""

    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_vhost: str = "/"

    rabbitmq_frame_exchange: str = "vigilante.frames"
    rabbitmq_frame_routing_key: str = "frame.ingested"
    rabbitmq_frame_queue_name: str = "vigilante.recognition.frame_ingested"
    rabbitmq_frame_dlx: str = "vigilante.frames.dlx"
    rabbitmq_frame_dlq: str = "vigilante.recognition.frame_ingested.dlq"
    rabbitmq_frame_dlq_routing_key: str = "frame.ingested.dlq"
    rabbitmq_prefetch_count: int = 10
    rabbitmq_retry_limit: int = 3
    rabbitmq_idle_timeout_seconds: float = 1.0
    recognition_event_exchange: str = "vigilante.recognition.events"

    observation_window_seconds: int = 15
    presence_confirmation_frames: int = 3
    face_quality_threshold: float = 0.75
    face_match_threshold: float = 0.82
    second_best_margin: float = 0.05
    face_backend: str = "simple"
    insightface_enabled: bool = True
    insightface_model_name: str = "buffalo_l"
    insightface_provider: str = "cpu"
    insightface_model_root: str = ""
    insightface_det_size: str = "640,640"
    insightface_detection_threshold: float = 0.5
    insightface_max_faces: int = 1
    insightface_min_face_bbox_size: int = 0
    insightface_min_face_area_ratio: float = 0.0
    insightface_camera_overrides_json: str = ""
    insightface_camera_metrics_log_every_n_frames: int = 25
    embedding_backend: str = "simple_face_crop_512"
    known_face_gallery_path: str = "app/data/dev_known_face_gallery.json"
    cross_camera_match_threshold: float = 0.85
    cross_camera_time_window_seconds: int = 600
    identity_conflict_margin: float = 0.25
    manual_review_threshold: float = 0.35
    semantic_descriptor_backend: str = "qwen_vl"
    semantic_use_real_vlm: bool = False
    semantic_vlm_primary_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    semantic_vlm_fallback_model: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    semantic_device: str = "auto"
    semantic_timeout_seconds: int = 45
    semantic_enable_fallback: bool = True
    semantic_similarity_threshold: float = 0.72
    recurrent_subject_threshold: float = 0.78
    case_suggestion_threshold: float = 0.9
    ingestion_jsonl_path: str = "../vigilante-ingestion/outbox/frame_ingested.jsonl"
    ingestion_frame_search_roots: str = ""
    ingestion_checkpoint_path: str = ".runtime/ingestion/checkpoints.json"
    ingestion_deduper_path: str = ".runtime/ingestion/processed_events.json"
    ingestion_rejected_events_path: str = ".runtime/ingestion/rejected_events.jsonl"
    ingestion_track_continuity_window_seconds: int = 15
    storage_s3_endpoint: str = "localhost:9000"
    storage_s3_access_key: str = "minio"
    storage_s3_secret_key: str = "minio123"
    storage_s3_secure: bool = False
    storage_s3_region: str | None = None
    storage_s3_cache_dir: str = ".runtime/ingestion/frame-cache"
    storage_s3_connect_timeout_seconds: float = 3.0
    storage_s3_read_timeout_seconds: float = 15.0
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def sqlalchemy_url(self) -> str:
        password = self.db_password or ""
        return (
            f"postgresql+psycopg://{self.db_user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def ingestion_frame_search_root_paths(self) -> list[Path]:
        return [
            Path(raw_path.strip())
            for raw_path in self.ingestion_frame_search_roots.split(",")
            if raw_path.strip()
        ]


settings = Settings()
