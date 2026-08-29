"""Groq client wrapper: disk-cached, retried, cost-metered.

Every call is keyed on sha256(system, user, model, kwargs) and cached to disk
under config.llm.cache_dir. A cache hit costs nothing and needs no key — this
is what lets `make repro` regenerate every narrator/grounding number from a
clean checkout with no GROQ_API_KEY set (build spec 10.0). A cache miss with
no key raises NoGroqKey, which narrator.py catches to fall back to the
deterministic template.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


class NoGroqKey(RuntimeError):
    """Raised on a cache miss with no GROQ_API_KEY available."""


def _cache_key(system: str, user: str, model: str, kw: dict) -> str:
    payload = json.dumps([system, user, model, kw], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def complete(
    system: str,
    user: str,
    model: str,
    cache_dir: Path,
    max_retries: int = 5,
    retry_base_delay_sec: float = 1.0,
    request_timeout_sec: float = 60.0,
    **kw,
) -> tuple[str, dict | None]:
    """Return (content, usage_dict). Reads/writes the disk cache in cache_dir.

    kw is passed straight through to the Groq chat.completions.create call
    (temperature, max_completion_tokens, reasoning_effort, ...) and folded
    into the cache key so a config change invalidates stale cache entries.
    """
    cache_dir = Path(cache_dir)
    key = _cache_key(system, user, model, kw)
    hit = cache_dir / f"{key}.json"

    if hit.exists():
        cached = json.loads(hit.read_text())
        return cached["content"], cached.get("usage")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise NoGroqKey(
            f"Cache miss for key {key[:12]}... and GROQ_API_KEY is not set. "
            "Set it to call Groq live, or accept the template fallback."
        )

    from groq import Groq  # imported lazily so a missing key never requires the SDK

    client = Groq(api_key=api_key, timeout=request_timeout_sec)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kw,
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry any transient SDK/network error
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            delay = retry_base_delay_sec * (2**attempt)
            log.warning("Groq call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, max_retries, exc, delay)
            time.sleep(delay)
    else:  # pragma: no cover - loop always breaks or raises
        raise last_exc  # type: ignore[misc]

    content = r.choices[0].message.content
    usage = r.usage.model_dump() if r.usage else None

    cache_dir.mkdir(parents=True, exist_ok=True)
    hit.write_text(json.dumps({
        "content": content,
        "model": model,
        "usage": usage,
    }))
    return content, usage
