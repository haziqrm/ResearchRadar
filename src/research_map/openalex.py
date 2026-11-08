"""Small, dependency-light client for the OpenAlex API.

The API key is sent in an Authorization header so it does not appear in URLs,
proxy logs, or exception messages.  Use :meth:`OpenAlexClient.from_env` in the
application; it loads the repository's ``.env`` file automatically.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


OPENALEX_BASE_URL = "https://api.openalex.org"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class OpenAlexError(RuntimeError):
    """Raised when OpenAlex returns an unsuccessful response."""


class OpenAlexClient:
    """Authenticated client for OpenAlex works and usage information."""

    def __init__(
        self,
        api_key: str,
        *,
        email: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("An OpenAlex API key is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.api_key = api_key.strip()
        self.email = email.strip() if email else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()

    @classmethod
    def from_env(
        cls,
        env_file: str | Path = DEFAULT_ENV_FILE,
        **kwargs: Any,
    ) -> "OpenAlexClient":
        """Build a client from ``OPENALEX_API_KEY`` and ``OPENALEX_EMAIL``.

        Existing process environment variables take precedence over values in
        the file, which is useful in CI and deployed environments.
        """
        load_dotenv(dotenv_path=env_file, override=False)
        api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENALEX_API_KEY is not set; add it to the repository .env file"
            )
        return cls(api_key, email=os.getenv("OPENALEX_EMAIL"), **kwargs)

    def _get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{OPENALEX_BASE_URL}/{path.lstrip('/')}"
        query = dict(params or {})
        if self.email:
            query.setdefault("mailto", self.email)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "ResearchRadar/0.1",
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url, params=query, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise OpenAlexError(f"OpenAlex request failed: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 2**attempt
                    except ValueError:
                        delay = 2**attempt
                    time.sleep(max(0.0, delay))
                    continue

            if not response.ok:
                try:
                    detail = response.json().get("message", response.text)
                except (ValueError, AttributeError):
                    detail = response.text
                raise OpenAlexError(
                    f"OpenAlex returned HTTP {response.status_code}: {detail}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise OpenAlexError("OpenAlex returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise OpenAlexError("OpenAlex returned an unexpected response shape")
            return payload

        raise AssertionError("retry loop exited unexpectedly")

    def get_work(
        self,
        work_id: str,
        *,
        select: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return one work by an OpenAlex ID, DOI, PMID, or MAG ID."""
        if not work_id.strip():
            raise ValueError("work_id cannot be empty")
        params = {"select": ",".join(select)} if select else None
        return self._get(f"works/{work_id.strip()}", params)

    def search_works(
        self,
        query: str,
        *,
        filters: str | None = None,
        per_page: int = 100,
        cursor: str | None = None,
        select: Sequence[str] | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Search works and return OpenAlex's ``meta``/``results`` envelope."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100")

        params: dict[str, Any] = {"search": query.strip(), "per_page": per_page}
        if filters:
            params["filter"] = filters
        if cursor:
            params["cursor"] = cursor
        if select:
            params["select"] = ",".join(select)
        if sort:
            params["sort"] = sort
        return self._get("works", params)

    def iter_works(
        self,
        query: str,
        *,
        filters: str | None = None,
        per_page: int = 100,
        max_results: int | None = None,
        select: Sequence[str] | None = None,
        sort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield matching works using key-enabled cursor pagination."""
        if max_results is not None and max_results < 0:
            raise ValueError("max_results cannot be negative")
        if max_results == 0:
            return

        cursor: str | None = "*"
        yielded = 0
        while cursor:
            page = self.search_works(
                query,
                filters=filters,
                per_page=per_page,
                cursor=cursor,
                select=select,
                sort=sort,
            )
            results = page.get("results", [])
            if not isinstance(results, list):
                raise OpenAlexError("OpenAlex response has no results list")
            for work in results:
                if not isinstance(work, dict):
                    raise OpenAlexError("OpenAlex returned an invalid work record")
                yield work
                yielded += 1
                if max_results is not None and yielded >= max_results:
                    return
            if not results:
                return
            meta = page.get("meta", {})
            cursor = meta.get("next_cursor") if isinstance(meta, dict) else None

    def sample_works(
        self,
        filters: str,
        n: int,
        *,
        seed: int = 0,
        select: Sequence[str] | None = None,
        per_page: int = 100,
    ) -> Iterator[dict[str, Any]]:
        """Yield a seeded random sample of up to ``n`` works matching ``filters``.

        The same filter and seed give the same sample while the OpenAlex index
        is unchanged.  OpenAlex caps a sample at 10,000 works.
        """
        if not filters.strip():
            raise ValueError("filters cannot be empty")
        if not 1 <= n <= 10_000:
            raise ValueError("n must be between 1 and 10,000")
        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100")

        params: dict[str, Any] = {
            "filter": filters.strip(), "sample": n, "seed": seed, "per_page": per_page,
        }
        if select:
            params["select"] = ",".join(select)
        yielded = 0
        page = 1
        while yielded < n:
            payload = self._get("works", {**params, "page": page})
            results = payload.get("results", [])
            if not isinstance(results, list):
                raise OpenAlexError("OpenAlex response has no results list")
            if not results:
                return
            for work in results:
                if not isinstance(work, dict):
                    raise OpenAlexError("OpenAlex returned an invalid work record")
                yield work
                yielded += 1
                if yielded >= n:
                    return
            page += 1

    def count_works_by(self, filters: str, group_by: str) -> dict[str, int]:
        """Count works matching ``filters`` per value of ``group_by`` (e.g. year)."""
        if not group_by.strip():
            raise ValueError("group_by cannot be empty")
        params: dict[str, Any] = {"group_by": group_by.strip()}
        if filters.strip():
            params["filter"] = filters.strip()
        groups = self._get("works", params).get("group_by")
        if not isinstance(groups, list):
            raise OpenAlexError("OpenAlex response has no group_by list")
        return {
            str(group["key"]): int(group["count"])
            for group in groups
            if isinstance(group, dict) and "key" in group and "count" in group
        }

    def get_works(self, work_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Fetch up to 100 works in one request using OpenAlex's OR filter."""
        ids = [work_id.strip() for work_id in work_ids if work_id.strip()]
        if not ids:
            return []
        if len(ids) > 100:
            raise ValueError("OpenAlex accepts at most 100 OR values per filter")
        payload = self._get(
            "works", {"filter": f"openalex_id:{'|'.join(ids)}", "per_page": len(ids)}
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise OpenAlexError("OpenAlex response has no results list")
        return results

    def get_rate_limit(self) -> dict[str, Any]:
        """Return the authenticated account's current API usage information."""
        return self._get("rate-limit")

