from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from app.config import settings
from app.logging import current_log_level_name, write_runtime_log_level
from app.services.runtime_metrics_summary_service import RuntimeMetricsSummaryService
from app.services.runtime_recommendation_service import RuntimeRecommendationService

logger = logging.getLogger(__name__)


class RuntimeMetricsHttpServer:
    def __init__(
        self,
        *,
        store: Any,
        host: str,
        port: int,
        summary_service: RuntimeMetricsSummaryService | None = None,
        recommendation_service: RuntimeRecommendationService | None = None,
    ) -> None:
        self.store = store
        self.host = host
        self.port = int(port)
        self.summary_service = summary_service or RuntimeMetricsSummaryService()
        self.recommendation_service = recommendation_service
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return

        store = self.store
        summary_service = self.summary_service
        recommendation_service = self.recommendation_service

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
                if self.path in {"/health", "/runtime-metrics/health"}:
                    self._write_json({"status": "ok"})
                    return
                if self.path == "/admin/log-level":
                    self._write_json(
                        {
                            "level": current_log_level_name(),
                            "runtime_path": settings.runtime_log_level_path,
                        }
                    )
                    return
                if self.path in {"/", "/runtime-metrics/summary"}:
                    summary = summary_service.summarize_store(store)
                    self._write_json(summary)
                    return
                if self.path == "/runtime-metrics/recommendations":
                    if recommendation_service is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    self._write_json(recommendation_service.generate(persist=False))
                    return
                if self.path == "/runtime-metrics/recommendations/cameras":
                    if recommendation_service is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    result = recommendation_service.generate(persist=False)
                    self._write_json(
                        {
                            "schema_version": result.get("schema_version"),
                            "rule_set_version": result.get("rule_set_version"),
                            "generated_at": result.get("generated_at"),
                            "window_summary": result.get("window_summary"),
                            "by_camera": result.get("by_camera", {}),
                            "recommendations": result.get("recommendations", []),
                        }
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
                if self.path != "/admin/log-level":
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8") if length > 0 else "{}")
                    level = write_runtime_log_level(
                        settings.runtime_log_level_path,
                        str(payload.get("level", "")),
                        source="admin_http",
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    self._write_json({"error": "invalid_log_level", "message": str(exc)}, status_code=422)
                    return
                self._write_json({"level": level, "runtime_path": settings.runtime_log_level_path})

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("runtime_metrics_http " + format, *args)

            def _write_json(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
                    "utf-8"
                )
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = Thread(
            target=self._server.serve_forever,
            name="runtime-metrics-http",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "runtime_metrics_http_started url=http://%s:%s/runtime-metrics/summary",
            self.host,
            self.port,
        )

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


def start_runtime_metrics_http_server(
    *,
    store: Any,
    host: str,
    port: int,
    recommendation_service: RuntimeRecommendationService | None = None,
) -> RuntimeMetricsHttpServer | None:
    server = RuntimeMetricsHttpServer(
        store=store,
        host=host,
        port=port,
        recommendation_service=recommendation_service,
    )
    try:
        server.start()
    except OSError as exc:
        logger.warning(
            "runtime_metrics_http_start_failed host=%s port=%s error=%s",
            host,
            port,
            exc,
        )
        return None
    return server
