from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from app.services.runtime_metrics_summary_service import RuntimeMetricsSummaryService

logger = logging.getLogger(__name__)


class RuntimeMetricsHttpServer:
    def __init__(
        self,
        *,
        store: Any,
        host: str,
        port: int,
        summary_service: RuntimeMetricsSummaryService | None = None,
    ) -> None:
        self.store = store
        self.host = host
        self.port = int(port)
        self.summary_service = summary_service or RuntimeMetricsSummaryService()
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return

        store = self.store
        summary_service = self.summary_service

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
                if self.path in {"/health", "/runtime-metrics/health"}:
                    self._write_json({"status": "ok"})
                    return
                if self.path in {"/", "/runtime-metrics/summary"}:
                    summary = summary_service.summarize_store(store)
                    self._write_json(summary)
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("runtime_metrics_http " + format, *args)

            def _write_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
                    "utf-8"
                )
                self.send_response(200)
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
) -> RuntimeMetricsHttpServer | None:
    server = RuntimeMetricsHttpServer(store=store, host=host, port=port)
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
