from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, StrEnum
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from prediction_desk.audit.hashing import GENESIS_HASH, canonical_json, compute_hash

TS = datetime(2024, 3, 12, 12, 30, 0, 123456, tzinfo=UTC)


class Side(Enum):
    YES = "yes"


class Kind(StrEnum):
    BUY = "buy"


class Ratio(Enum):
    HALF = 0.5


GOLDEN_ENVELOPE = {
    "actor": "system",
    "entity": "test",
    "entity_id": "1",
    "action": "noop",
    "payload": {"i": 1},
}
# Pinned once from the implementation and cross-checked with `shasum -a 256` over the raw bytes.
# Changing it means every stored chain stops verifying: treat as a format break, not a fixup.
GOLDEN_HASH = "b956a4d07efc6f039a0995f4de03242486b83b77d323715afc816a43acbf122c"


def test_canonical_json_encodes_rich_types_deterministically() -> None:
    payload = {
        "b": Decimal("0.12345"),
        "a": TS,
        "u": UUID(int=1),
        "s": Side.YES,
        "t": "CPI ≥ 0.3%",
    }
    assert (
        canonical_json(payload)
        == (
            '{"a":"2024-03-12T12:30:00.123456+00:00","b":"0.12345","s":"yes","t":"CPI ≥ 0.3%",'
            '"u":"00000000-0000-0000-0000-000000000001"}'
        ).encode()
    )


def test_canonical_json_sorts_nested_keys_and_keeps_native_types() -> None:
    payload = {"z": {"b": 1, "a": [True, None, 2]}, "k": Kind.BUY}
    assert canonical_json(payload) == b'{"k":"buy","z":{"a":[true,null,2],"b":1}}'


def test_canonical_json_encodes_date() -> None:
    assert canonical_json({"d": date(2024, 3, 12)}) == b'{"d":"2024-03-12"}'


def test_canonical_json_normalizes_datetime_to_utc() -> None:
    plus_two = datetime(2024, 3, 12, 14, 30, tzinfo=timezone(timedelta(hours=2)))
    assert canonical_json({"x": plus_two}) == b'{"x":"2024-03-12T12:30:00.000000+00:00"}'


def test_canonical_json_decimal_is_not_normalized() -> None:
    assert canonical_json({"x": Decimal("1.50")}) == b'{"x":"1.50"}'


@pytest.mark.parametrize("bad", [0.5, -0.0, 1e20, float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_floats(bad: float) -> None:
    with pytest.raises(TypeError, match="float"):
        canonical_json({"x": bad})


def test_canonical_json_rejects_nested_float() -> None:
    with pytest.raises(TypeError, match="float"):
        canonical_json({"x": [1, {"y": [2, 0.5]}]})
    with pytest.raises(TypeError, match="float"):
        canonical_json({"x": (1, 0.5)})


@pytest.mark.parametrize("key", [1, True, None])
def test_canonical_json_rejects_non_str_keys(key: object) -> None:
    with pytest.raises(TypeError, match="keys"):
        canonical_json({"a": {key: 1}})
    with pytest.raises(TypeError, match="keys"):
        canonical_json({"a": [{key: 1}]})
    with pytest.raises(TypeError, match="keys"):
        canonical_json({key: 1})  # type: ignore[dict-item]


def test_canonical_json_rejects_enum_with_float_value() -> None:
    with pytest.raises(TypeError, match="float"):
        canonical_json({"x": Ratio.HALF})


def test_compute_hash_golden_vector() -> None:
    ts = datetime(2024, 3, 12, 12, 0, tzinfo=UTC)
    assert compute_hash(GENESIS_HASH, GOLDEN_ENVELOPE, 1, ts) == GOLDEN_HASH


def test_compute_hash_takes_envelope_keyword() -> None:
    ts = datetime(2024, 3, 12, 12, 0, tzinfo=UTC)
    assert compute_hash(GENESIS_HASH, envelope=GOLDEN_ENVELOPE, seq=1, ts=ts) == GOLDEN_HASH


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_canonical_json_rejects_non_finite_decimal(bad: Decimal) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"x": bad})


def test_canonical_json_rejects_naive_datetime_and_unknown_types() -> None:
    with pytest.raises(ValueError, match="timezone"):
        canonical_json({"x": datetime(2024, 1, 1)})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


@given(st.dictionaries(st.text(), st.one_of(st.integers(), st.text(), st.booleans()), max_size=8))
def test_canonical_json_ignores_insertion_order(d: dict[str, object]) -> None:
    assert canonical_json(d) == canonical_json(dict(reversed(list(d.items()))))


def test_compute_hash_is_stable_and_sensitive() -> None:
    h = compute_hash(GENESIS_HASH, {"a": 1}, 1, TS)
    assert h == compute_hash(GENESIS_HASH, {"a": 1}, 1, TS) and len(h) == 64
    assert h != compute_hash(GENESIS_HASH, {"a": 2}, 1, TS)
    assert h != compute_hash(GENESIS_HASH, {"a": 1}, 2, TS)
    assert h != compute_hash(h, {"a": 1}, 1, TS)
    assert h != compute_hash(GENESIS_HASH, {"a": 1}, 1, TS + timedelta(microseconds=1))


def test_compute_hash_equal_instants_in_different_zones_match() -> None:
    plus_two = TS.astimezone(timezone(timedelta(hours=2)))
    assert compute_hash(GENESIS_HASH, {}, 1, TS) == compute_hash(GENESIS_HASH, {}, 1, plus_two)


def test_compute_hash_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone"):
        compute_hash(GENESIS_HASH, {}, 1, datetime(2024, 1, 1))


@pytest.mark.parametrize("bad", ["abc", "", "A" * 64, "g" * 64, "0" * 63, "0" * 64 + "\n"])
def test_compute_hash_rejects_malformed_prev_hash(bad: str) -> None:
    with pytest.raises(ValueError, match="prev_hash"):
        compute_hash(bad, {}, 1, TS)


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_compute_hash_rejects_non_int_seq(bad: object) -> None:
    with pytest.raises(TypeError):
        compute_hash(GENESIS_HASH, {}, bad, TS)  # type: ignore[arg-type]
