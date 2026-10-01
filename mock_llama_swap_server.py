#!/usr/bin/env python3
"""Minimal llama-swap-compatible mock endpoint for benchmark smoke tests."""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence


class MockLlamaSwapHandler(BaseHTTPRequestHandler):
    server_version = "MockLlamaSwap/1.0"

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _extract_user_text(self, payload: object) -> str:
        messages = payload.get("messages") if isinstance(payload, dict) else []
        if not isinstance(messages, list):
            return ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                            elif isinstance(item.get("input_audio"), dict):
                                parts.append(str(item["input_audio"].get("data", "")))
                        elif isinstance(item, str):
                            parts.append(item)
                    return "\n".join(parts)
        return ""

    def _response_text(self, payload: object) -> str:
        text = self._extract_user_text(payload)
        lowered = text.lower()

        if "llama-swap-ready" in lowered:
            return "LLAMA-SWAP-READY"

        code = re.search(r"ORCHID-[A-Z0-9-]+", text)
        if code:
            return code.group(0)

        if "df -h" in lowered:
            return "df -h shows the filesystem usage summary and which mount is closest to full."
        if "len(items) - 1" in lowered or "range(len(items)" in lowered:
            return "Use range(len(items) - 1) so the last valid pair is pairs[-1], without indexing past the end."
        if "technology" in lowered:
            return "technology"
        if "2024" in lowered:
            return "2024"
        if "negative" in lowered:
            return "negative"
        if "positive" in lowered:
            return "positive"

        if "benchmark" in lowered and "jsonl" in lowered:
            return "benchmark harness uses JSONL logs for each run."
        if "file" in lowered and "compress" in lowered:
            return "file compression is handled before upload."

        return "mock llama-swap response"

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            models = [{"id": model_name} for model_name in self.server.models]
            self._send_json(200, {"data": models})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.startswith("/v1/chat/completions"):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return

            text = self._response_text(payload)
            model = str(payload.get("model") or self.server.models[0])
            created = int(time.time())
            chunks = [text[i : i + 12] for i in range(0, len(text), 12)] or [text]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            for index, chunk in enumerate(chunks):
                event = {
                    "id": f"chatcmpl-{created}-{index}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                }
                self.wfile.write(f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode("utf-8"))
                self.wfile.flush()

            final_event = {
                "id": f"chatcmpl-{created}-final",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 128, "completion_tokens": max(1, len(text.split()))},
            }
            self.wfile.write(f"data: {json.dumps(final_event, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._send_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:
        return


def run_server(host: str, port: int, models: Sequence[str]) -> None:
    server = ThreadingHTTPServer((host, port), MockLlamaSwapHandler)
    server.models = list(models)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a lightweight llama-swap mock server used by local benchmarks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()
    models = args.model or ["mock-model"]
    run_server(args.host, args.port, models)
