"""HTTP client helper with retry, backoff, and endpoint fallback support."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from http.cookiejar import CookieJar
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


@dataclass(frozen=True)
class HTTPRetryConfig:
    retries: int = 4
    backoff_seconds: float = 0.8
    max_backoff_seconds: float = 8.0
    jitter_seconds: float = 0.25
    timeout_seconds: int = 20


class RetryingHTTPClient:
    def __init__(self, retry_config: HTTPRetryConfig | None = None) -> None:
        self.retry_config = retry_config or HTTPRetryConfig()
        self._cookie_jar = CookieJar()
        self._opener = build_opener(HTTPCookieProcessor(self._cookie_jar))
        self._bootstrapped_hosts: set[str] = set()

    def get_json(
        self,
        urls: list[str],
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        bootstrap_url: str | None = None,
    ) -> dict:
        if not urls:
            raise ValueError("at least one URL is required")

        q = ""
        if params:
            q = urlencode(params)

        merged_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }
        if headers:
            merged_headers.update(headers)

        if bootstrap_url:
            self._bootstrap(bootstrap_url, merged_headers)
            merged_headers.setdefault("Referer", bootstrap_url)

        last_err: Exception | None = None
        for base_url in urls:
            url = f"{base_url}?{q}" if q else base_url
            try:
                return self._get_json_with_retry(url, merged_headers)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last_err = exc
                continue

        if last_err is None:
            raise RuntimeError("failed to fetch from all endpoint candidates")
        raise RuntimeError(f"all endpoint candidates failed: {last_err}") from last_err

    def _bootstrap(self, url: str, headers: dict[str, str]) -> None:
        host = _host_of(url)
        if host in self._bootstrapped_hosts:
            return
        req = Request(url, headers=headers)
        with self._opener.open(req, timeout=self.retry_config.timeout_seconds):
            pass
        self._bootstrapped_hosts.add(host)

    def _get_json_with_retry(self, url: str, headers: dict[str, str]) -> dict:
        for attempt in range(self.retry_config.retries + 1):
            try:
                req = Request(url, headers=headers)
                with self._opener.open(req, timeout=self.retry_config.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError("non-dict payload", body, 0)
                return parsed
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
                if attempt >= self.retry_config.retries:
                    raise
                sleep_s = min(
                    self.retry_config.max_backoff_seconds,
                    self.retry_config.backoff_seconds * (2**attempt),
                )
                sleep_s += random.uniform(0.0, self.retry_config.jitter_seconds)
                time.sleep(sleep_s)
        raise RuntimeError("unreachable retry loop")


def _host_of(url: str) -> str:
    marker = "://"
    idx = url.find(marker)
    if idx == -1:
        return url.split("/", maxsplit=1)[0]
    host_part = url[idx + len(marker) :]
    return host_part.split("/", maxsplit=1)[0]
