"""Canonical JSON bytes and the chained SHA-256 for the append-only audit log."""

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from prediction_desk.domain.types import require_utc

GENESIS_HASH: str = "0" * 64

_HASH_RE = re.compile(r"[0-9a-f]{64}")


def _iso(ts: datetime) -> str:
    return require_utc(ts).isoformat(timespec="microseconds")


def _default(obj: object) -> Any:
    # datetime is checked before date because datetime subclasses date.
    if isinstance(obj, Decimal):
        if not obj.is_finite():
            raise ValueError(f"non-finite Decimal is not allowed in audit payloads: {obj!r}")
        return str(obj)
    if isinstance(obj, datetime):
        return _iso(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _reject_unstable(value: object) -> None:
    """Floats and non-str keys do not survive a JSONB round trip (1e20, -0.0, int keys)."""
    if isinstance(value, float):
        raise TypeError("float is not allowed in audit payloads; use Decimal or str")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"audit payload keys must be str, got {type(key).__name__}")
            _reject_unstable(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_unstable(item)


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Deterministic UTF-8 JSON: sorted keys, no whitespace, no floats, str keys only."""
    _reject_unstable(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    ).encode("utf-8")


def compute_hash(prev_hash: str, payload: Mapping[str, Any], seq: int, ts: datetime) -> str:
    """sha256(prev_hash || canonical_json(payload) || seq || ts), hex-encoded."""
    if not _HASH_RE.fullmatch(prev_hash):
        raise ValueError("prev_hash must be 64 lowercase hex characters")
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise TypeError(f"seq must be int, got {type(seq).__name__}")
    digest = hashlib.sha256()
    digest.update(prev_hash.encode())
    digest.update(canonical_json(payload))
    digest.update(str(seq).encode())
    digest.update(_iso(ts).encode())
    return digest.hexdigest()
