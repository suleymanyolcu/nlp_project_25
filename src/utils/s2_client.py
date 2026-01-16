from __future__ import annotations

import os
import time
import random
import requests
from typing import Any, Dict, Optional


class SemanticScholarClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        sleep_seconds: float = 1.1,
        max_retries: int = 6,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("S2_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set environment variable S2_API_KEY or pass api_key=..."
            )
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.timeout = timeout

    @property
    def headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key}

    def get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rate-limited GET with retries.
        Respects global 1 request/sec by sleeping after each request attempt.
        """
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)

                # Always sleep after an attempt to respect 1 req/sec globally
                time.sleep(self.sleep_seconds)

                # Retry on rate-limit or server errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(30.0, (2 ** (attempt - 1)) + random.random())
                    time.sleep(wait)
                    continue

                # Surface useful debug info on 4xx
                if resp.status_code >= 400:
                    snippet = resp.text[:500]
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code} for {resp.url}\nResponse (truncated): {snippet}"
                    )

                return resp.json()

            except Exception as e:
                last_err = e
                # backoff for unexpected transient network issues
                wait = min(30.0, (2 ** (attempt - 1)) + random.random())
                time.sleep(wait)

        raise RuntimeError(f"Semantic Scholar request failed after retries. Last error: {last_err}")
