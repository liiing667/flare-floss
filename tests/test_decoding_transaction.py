# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from floss.const import DS_MAX_DIFF_BYTES, DS_MAX_CONTEXT_RETRIES
from floss.results import AddressType
from floss.string_decoder import extract_delta_bytes
from floss.decoding_manager import Delta, Snapshot, EmulationResult
from floss.decoding_transaction import (
    EvidenceKey,
    TransactionStatus,
    DecodingTransaction,
    TransactionExhausted,
    truncate_diff,
)

FVA = 0x6000
DECODED_AT = 0x5000
BASE = 0x1000


def make_snapshot(buf: bytes, base: int = BASE, sp: int = 0x7FFF0000, pc: int = 0x4000) -> Snapshot:
    memory_map = (base, base + len(buf), (base, len(buf), 7, "test"), buf)
    return Snapshot([memory_map], sp, pc)


def make_delta(before: bytes, after: bytes, base: int = BASE) -> Delta:
    assert len(before) == len(after)
    return Delta(make_snapshot(before, base=base), make_snapshot(after, base=base))


def make_result(deltas, completed=True, instructions=10):
    return EmulationResult(deltas=list(deltas), completed=completed, instructions_executed=instructions)


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------


def test_truncate_diff_leaves_short_diffs_untouched():
    data = b"Hello world\x00"
    kept, truncation = truncate_diff(data, BASE)
    assert kept == data
    assert truncation.truncated is False
    assert truncation.original_length == truncation.kept_length == len(data)


def test_truncate_diff_caps_overlong_diffs():
    data = b"A" * (DS_MAX_DIFF_BYTES * 2)
    kept, truncation = truncate_diff(data, BASE)
    assert len(kept) == DS_MAX_DIFF_BYTES
    assert truncation.truncated is True
    assert truncation.original_length == len(data)
    assert truncation.kept_length == DS_MAX_DIFF_BYTES


def test_truncate_diff_keeps_even_length_for_utf16():
    # an odd cap would leave a dangling UTF-16 trailing byte
    odd_cap = DS_MAX_DIFF_BYTES + 1
    data = b"A\x00" * odd_cap
    kept, truncation = truncate_diff(data, BASE, max_bytes=odd_cap)
    assert len(kept) % 2 == 0
    assert len(kept) == odd_cap - 1
    assert truncation.kept_length % 2 == 0


def test_extract_delta_bytes_returns_raw_overlong_diff():
    before = b"\x00" * (DS_MAX_DIFF_BYTES + 16)
    after = b"B" * (DS_MAX_DIFF_BYTES + 16)
    delta_bytes = extract_delta_bytes(make_delta(before, after), DECODED_AT, FVA)
    assert len(delta_bytes) == 1
    db = delta_bytes[0]
    assert db.address == BASE
    # the extractor itself does not truncate; truncation happens on admission
    assert len(db.bytes) == DS_MAX_DIFF_BYTES + 16
    assert db.was_truncated is False


def test_transaction_truncates_overlong_diff_on_admission():
    before = b"\x00" * (DS_MAX_DIFF_BYTES + 16)
    after = b"B" * (DS_MAX_DIFF_BYTES + 16)
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000)
    tx.begin_attempt()

    admitted = tx.record_emulation(make_result([make_delta(before, after)]), extract_delta_bytes)
    assert len(admitted) == 1
    assert len(admitted[0].bytes) == DS_MAX_DIFF_BYTES
    assert admitted[0].was_truncated is True
    assert admitted[0].truncation.original_length == DS_MAX_DIFF_BYTES + 16
    assert len(tx.truncations) == 1
    assert tx.truncations[0].kept_length == DS_MAX_DIFF_BYTES


def test_extract_delta_bytes_short_diff_not_truncated():
    before = b"\x00" * 16
    after = b"Hello FLOSS\x00\x00\x00\x00\x00"
    delta_bytes = extract_delta_bytes(make_delta(before, after), DECODED_AT, FVA)
    assert len(delta_bytes) == 1
    assert delta_bytes[0].was_truncated is False
    # trailing zero bytes are not part of the differing region
    assert delta_bytes[0].bytes == b"Hello FLOSS"


# ---------------------------------------------------------------------------
# retry / duplicate evidence
# ---------------------------------------------------------------------------


def test_successful_attempt_commits_in_single_try():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000)
    assert tx.status == TransactionStatus.PENDING

    tx.begin_attempt()
    result = make_result([make_delta(b"\x00" * 16, b"Hello FLOSS\x00\x00\x00\x00\x00")])
    admitted = tx.record_emulation(result, extract_delta_bytes)
    assert len(admitted) == 1

    tx.note_evidence(BASE, "Hello FLOSS")
    assert tx.commit() == TransactionStatus.COMMITTED
    assert tx.attempts == 1
    assert tx.succeeded is True
    assert EvidenceKey(BASE, "Hello FLOSS") in tx.committed_evidence


def test_retry_rolls_back_pending_state_without_losing_evidence():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000, max_retries=2)

    # first attempt aborts without producing any usable difference
    tx.begin_attempt()
    assert tx.record_emulation(make_result([], completed=False), extract_delta_bytes) == []
    assert tx.commit() == TransactionStatus.ROLLED_BACK
    assert tx.status == TransactionStatus.ROLLED_BACK
    assert tx.pending_delta_bytes == []
    assert tx.can_retry is True

    # the retry starts from the same context and now succeeds
    tx.begin_attempt()
    tx.record_emulation(make_result([make_delta(b"\x00" * 16, b"first string\x00\x00\x00\x00")]), extract_delta_bytes)
    tx.note_evidence(BASE, "first string")
    assert tx.commit() == TransactionStatus.COMMITTED
    assert tx.attempts == 2
    assert tx.committed_evidence == {EvidenceKey(BASE, "first string")}


def test_retry_does_not_duplicate_the_same_evidence():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000, max_retries=2)

    # first attempt is aborted by the instruction budget but had already
    # decoded a usable region; the evidence is retained instead of being lost
    tx.begin_attempt()
    tx.record_emulation(
        make_result([make_delta(b"\x00" * 16, b"repeat me!!\x00\x00\x00\x00\x00")], completed=False),
        extract_delta_bytes,
    )
    tx.note_evidence(BASE, "repeat me!!")
    assert tx.commit() == TransactionStatus.COMMITTED

    # only one unique piece of evidence is ever committed
    assert tx.committed_evidence == {EvidenceKey(BASE, "repeat me!!")}
    # and a hypothetical retry recognizes it as a duplicate
    assert tx.is_duplicate_evidence(BASE, "repeat me!!") is True


def test_transaction_is_exhausted_after_retry_budget():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000, max_retries=1)

    for expected_attempt in (1, 2):
        tx.begin_attempt()
        assert tx.attempts == expected_attempt
        # every attempt aborts without producing any difference
        result = make_result([], completed=False, instructions=5)
        assert tx.record_emulation(result, extract_delta_bytes) == []
        assert tx.commit() == TransactionStatus.ROLLED_BACK

    assert tx.can_retry is False
    assert tx.succeeded is False
    with pytest.raises(TransactionExhausted):
        tx.begin_attempt()


def test_instruction_budget_is_shared_across_attempts():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=100, max_retries=5)

    tx.begin_attempt()
    remaining = tx.consume_budget(60)
    assert remaining == 40
    assert tx.budget_exhausted is False
    tx.rollback("test")

    tx.begin_attempt()
    # the budget is not refreshed by a rollback/retry
    assert tx.instructions_consumed == 60
    assert tx.consume_budget(40) == 0
    assert tx.budget_exhausted is True


# ---------------------------------------------------------------------------
# empty diffs
# ---------------------------------------------------------------------------


def test_empty_difference_is_not_admitted():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000)
    tx.begin_attempt()

    identical = make_delta(b"\x00" * 32, b"\x00" * 32)
    admitted = tx.record_emulation(make_result([identical]), extract_delta_bytes)
    assert admitted == []
    assert tx.has_pending_evidence is False


def test_all_zero_difference_is_ignored():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000)
    tx.begin_attempt()

    zero_diff = make_delta(b"\x00" * 32, b"\x00" * 32)
    result = make_result([zero_diff], completed=True)
    assert tx.record_emulation(result, extract_delta_bytes) == []
    # a clean return with no differences is still a successful attempt
    assert tx.commit() == TransactionStatus.COMMITTED
    assert tx.succeeded is True


def test_aborted_empty_attempt_is_rolled_back_then_retried():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000, max_retries=2)

    tx.begin_attempt()
    tx.record_emulation(make_result([], completed=False), extract_delta_bytes)
    assert tx.commit() == TransactionStatus.ROLLED_BACK
    assert tx.can_retry is True

    # retry now produces a real string and commits
    tx.begin_attempt()
    tx.record_emulation(
        make_result([make_delta(b"\x00" * 16, b"recovered!!!\x00\x00\x00\x00")]),
        extract_delta_bytes,
    )
    tx.note_evidence(BASE, "recovered!!!")
    assert tx.commit() == TransactionStatus.COMMITTED
    assert tx.succeeded is True


def test_default_retry_constant_is_used():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=10)
    assert tx.max_retries == DS_MAX_CONTEXT_RETRIES
    assert DS_MAX_DIFF_BYTES % 2 == 0


def test_overlapping_deltas_are_all_admitted_within_attempt():
    # region-level dedup is intentionally avoided: the same buffer may be
    # re-decoded into different content between two imported API calls.
    # String-level dedup (via the evidence set / seen) handles true repeats.
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000)
    tx.begin_attempt()

    delta = make_delta(b"\x00" * 16, b"same evidence\x00\x00\x00")
    # the API-call delta and the final delta both describe the same region
    result = make_result([delta, Delta(delta.pre, delta.post)])
    admitted = tx.record_emulation(result, extract_delta_bytes)
    assert len(admitted) == 2
    assert all(db.address_type == AddressType.GLOBAL for db in admitted)


def test_orchestration_failure_then_retry_reports_each_string_once():
    """mirror the retry loop used by floss.string_decoder.decode_strings"""
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=20000, max_retries=2)
    reported = []

    # attempt 1: emulation escapes without any difference
    tx.begin_attempt()
    tx.consume_budget(20000)
    tx.record_emulation(make_result([], completed=False, instructions=20000), extract_delta_bytes)
    assert tx.commit() == TransactionStatus.ROLLED_BACK

    # attempt 2: same context is retried and now decodes a string
    assert tx.can_retry is True
    tx.begin_attempt()
    delta_bytes_list = tx.record_emulation(
        make_result([make_delta(b"\x00" * 16, b"recovered!!!\x00\x00\x00\x00")]),
        extract_delta_bytes,
    )
    for delta_bytes in delta_bytes_list:
        if not tx.is_duplicate_evidence(delta_bytes.address, "recovered!!!"):
            tx.note_evidence(delta_bytes.address, "recovered!!!")
            reported.append((delta_bytes.address, "recovered!!!"))

    assert tx.commit() == TransactionStatus.COMMITTED
    assert tx.succeeded is True
    assert reported == [(BASE, "recovered!!!")]
    assert len(tx.committed_evidence) == 1


def test_orchestration_budget_exhaustion_prevents_further_attempts():
    tx = DecodingTransaction(FVA, DECODED_AT, instruction_budget=100, max_retries=5)

    tx.begin_attempt()
    tx.consume_budget(100)
    tx.record_emulation(make_result([], completed=False, instructions=100), extract_delta_bytes)
    tx.commit()

    # the shared budget is exhausted, mirroring the guard in decode_strings
    if tx.budget_exhausted:
        tx.rollback("shared instruction budget exhausted")

    assert tx.budget_exhausted is True
    assert tx.succeeded is False
    # the transaction still offers its retry slot, but the caller's budget
    # guard prevents starting another expensive emulation
    assert tx.can_retry is True
