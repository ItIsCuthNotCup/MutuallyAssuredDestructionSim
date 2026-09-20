"""Thin async client for TypeSafe's System One endpoint (model: Jev)."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


class JevClient:
    def __init__(self, api_key: str | None = None, concurrency: int = 16, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("Jev")
        if not self.api_key:
            raise RuntimeError("Set TYPESAFE_API_KEY (or Jev) in the environment")
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(timeout=timeout)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        body = {"state": state, "model": MODEL, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with self._sem:
            for attempt in range(4):
                try:
                    r = await self._client.post(API_URL, json=body, headers=headers)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.HTTPStatusError(r.text, request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    self.calls += 1
                    usage = data.get("usage", {})
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                    return data["answers"]
                except (httpx.HTTPStatusError, httpx.TransportError):
                    if attempt == 3:
                        raise
                    await asyncio.sleep(1.5 * 2**attempt)
        raise RuntimeError("unreachable")

    async def close(self) -> None:
        await self._client.aclose()
