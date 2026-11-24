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

"""
Resumable transactions for emulating string decoding contexts.

A :class:`DecodingTransaction` couples the four pieces of state that used to
live independently:

* context admission (which call context is being decoded),
* the instruction budget consumed while emulating that context,
* the snapshot differences produced by emulation, and
* the strings reported from those differences.

State produced by an attempt is only *pending* until :meth:`commit` is
called. If an attempt fails (emulation escaped, the instruction budget was
exhausted or only unusable partial bytes were produced) the pending state is
rolled back and the same context can be retried from its original snapshot.
Evidence that was committed by an earlier, partially successful attempt is
remembered, so a retry never reports the same decoded string twice.
"""

from enum import Enum
from typing import Set, List, Tuple, Callable, Optional
from dataclasses import field, dataclass

import floss.logging_
from floss.const import DS_MAX_DIFF_BYTES, DS_MAX_CONTEXT_RETRIES
from floss.results import AddressType
from floss.decoding_manager import Delta, EmulationResult

logger = floss.logging_.getLogger(__name__)

#: signature of the injected snapshot-diff extractor,
#: see :func:`floss.string_decoder.extract_delta_bytes`
DeltaExtractor = Callable[[Delta, int, int], List["DeltaBytes"]]


class TransactionStatus(str, Enum):
    #: admitted but no attempt has started yet
    PENDING = "pending"
    #: an attempt is in progress, state is not yet committed
    ACTIVE = "active"
    #: at least one attempt committed usable evidence (or cleanly produced none)
    COMMITTED = "committed"
    #: all attempts failed and the transaction has been rolled back
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class Truncation:
    """
    Records that a memory difference was shortened when it was admitted.

    Attributes:
        address: virtual address at which the difference starts
        original_length: length of the difference in the snapshot
        kept_length: length retained after truncation
    """

    address: int
    original_length: int
    kept_length: int

    @property
    def truncated(self) -> bool:
        return self.original_length > self.kept_length


@dataclass
class DeltaBytes:
    """
    The bytes of a single admitted (and possibly truncated) memory difference.
    """

    address: int
    address_type: AddressType
    bytes: bytes
    decoded_at: int
    decoding_routine: int
    truncation: Optional[Truncation] = None

    @property
    def was_truncated(self) -> bool:
        return self.truncation is not None and self.truncation.truncated


@dataclass(frozen=True)
class EvidenceKey:
    """
    Unique identity of a piece of committed decoding evidence.

    The address is part of the key because the same string may legitimately be
    decoded into distinct buffers (e.g. on the stack and on the heap).
    """

    address: int
    string: str


def truncate_diff(data: bytes, address: int, max_bytes: int = DS_MAX_DIFF_BYTES) -> Tuple[bytes, Truncation]:
    """
    Truncate a memory difference so it cannot carry an unbounded number of
    bytes across call contexts or retries.

    The retained length is rounded down to an even number of bytes, keeping
    UTF-16 code units intact instead of leaving a single trailing partial
    character behind. Differences shorter than ``max_bytes`` are returned
    unchanged.
    """
    original_length = len(data)
    if original_length <= max_bytes:
        return data, Truncation(address, original_length, original_length)

    kept_length = max_bytes - (max_bytes % 2)
    truncation = Truncation(address, original_length, kept_length)
    logger.debug(
        "truncating memory diff at 0x%x from %d to %d bytes",
        address,
        original_length,
        kept_length,
    )
    return data[:kept_length], truncation


class DecodingTransaction:
    """
    Resumable transaction for a single decoding call context.

    The transaction owns the instruction budget, the admitted snapshot
    differences and the reported strings. All three share one lifecycle:

    1. :meth:`begin_attempt` starts (or restarts) emulation of the context
       from its original snapshot; the instruction budget is shared across
       all attempts of the transaction and is not refreshed.
    2. :meth:`record_emulation` admits the deltas of an attempt, truncating
       over-long differences and dropping empty ones.
    3. :meth:`commit` publishes the evidence of the attempt; duplicates of
       already committed evidence are suppressed.
    4. :meth:`rollback` discards the pending state so :meth:`begin_attempt`
       can retry the same context without re-reporting anything.
    """

    def __init__(
        self,
        fva: int,
        decoded_at_va: int,
        instruction_budget: int,
        max_retries: int = DS_MAX_CONTEXT_RETRIES,
    ) -> None:
        self.fva = fva
        self.decoded_at_va = decoded_at_va
        self.instruction_budget = instruction_budget
        self.max_retries = max_retries

        self.status: TransactionStatus = TransactionStatus.PENDING
        self.attempts = 0
        self.instructions_consumed = 0

        #: evidence published by successful attempts, survives rollbacks
        self._committed: Set[EvidenceKey] = set()
        #: delta bytes published by successful attempts, survives rollbacks
        self._committed_delta_bytes: List[DeltaBytes] = []

        #: state belonging to the in-flight attempt, cleared on rollback
        self._pending_delta_bytes: List[DeltaBytes] = []
        self._pending_evidence: Set[EvidenceKey] = set()
        self._pending_completed = False
        self.truncations: List[Truncation] = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def begin_attempt(self) -> None:
        """Admit the context for another emulation attempt."""
        if not self.can_retry:
            raise TransactionExhausted(self.fva, self.decoded_at_va, self.attempts)
        self.attempts += 1
        self.status = TransactionStatus.ACTIVE
        # the context is restored from its original snapshot, but the
        # instruction budget is shared across the whole resumable transaction
        # and is not refreshed between attempts
        self._pending_delta_bytes = []
        self._pending_evidence = set()
        self._pending_completed = False

    def consume_budget(self, instructions: int) -> int:
        """Charge ``instructions`` to the attempt and return the remainder."""
        self.instructions_consumed += instructions
        return max(0, self.instruction_budget - self.instructions_consumed)

    @property
    def budget_exhausted(self) -> bool:
        return self.instructions_consumed >= self.instruction_budget

    @property
    def can_retry(self) -> bool:
        if self.status == TransactionStatus.COMMITTED:
            return False
        return self.attempts <= self.max_retries

    # ------------------------------------------------------------------
    # snapshot differences
    # ------------------------------------------------------------------
    def record_emulation(self, result: EmulationResult, extractor: DeltaExtractor) -> List[DeltaBytes]:
        """
        Admit the deltas of the current attempt.

        Over-long differences are truncated and empty differences are
        rejected. Returns the admitted delta bytes; decoded strings are
        de-duplicated separately via :meth:`is_duplicate_evidence`.
        """
        if self.status != TransactionStatus.ACTIVE:
            raise TransactionStateError("record_emulation called outside of an active attempt")

        self._pending_completed = result.completed

        admitted: List[DeltaBytes] = []

        for delta in result.deltas:
            for delta_bytes in extractor(delta, self.decoded_at_va, self.fva):
                kept, truncation = truncate_diff(delta_bytes.bytes, delta_bytes.address)
                if not kept:
                    # empty difference: nothing to report, do not retain it
                    continue

                if truncation.truncated:
                    self.truncations.append(truncation)

                admitted.append(
                    DeltaBytes(
                        address=delta_bytes.address,
                        address_type=delta_bytes.address_type,
                        bytes=kept,
                        decoded_at=delta_bytes.decoded_at,
                        decoding_routine=delta_bytes.decoding_routine,
                        truncation=truncation,
                    )
                )

        self._pending_delta_bytes = admitted
        return admitted

    @property
    def pending_delta_bytes(self) -> List[DeltaBytes]:
        return list(self._pending_delta_bytes)

    @property
    def has_pending_evidence(self) -> bool:
        return bool(self._pending_delta_bytes)

    # ------------------------------------------------------------------
    # final string report
    # ------------------------------------------------------------------
    def is_duplicate_evidence(self, address: int, string: str) -> bool:
        """Return True if the evidence was already published by an attempt."""
        key = EvidenceKey(address, string)
        return key in self._committed or key in self._pending_evidence

    def note_evidence(self, address: int, string: str) -> None:
        """Remember a string reported from the in-flight attempt."""
        self._pending_evidence.add(EvidenceKey(address, string))

    def commit(self) -> TransactionStatus:
        """
        Publish the pending attempt.

        A clean attempt that produced no difference (e.g. a decoding routine
        that legitimately writes nothing) is still a success. An aborted
        attempt without any usable evidence is rejected so it can be retried.
        """
        if self.status != TransactionStatus.ACTIVE:
            raise TransactionStateError("commit called outside of an active attempt")

        if not self.has_pending_evidence and not self._pending_completed:
            return self.rollback("emulation aborted without producing any memory difference")

        self._committed_delta_bytes.extend(self._pending_delta_bytes)
        self._committed.update(self._pending_evidence)
        self._pending_delta_bytes = []
        self._pending_evidence = set()
        self.status = TransactionStatus.COMMITTED
        logger.trace(
            "committed decoding transaction for 0x%x (call 0x%x) after %d attempt(s)",
            self.fva,
            self.decoded_at_va,
            self.attempts,
        )
        return self.status

    def rollback(self, reason: str = "") -> TransactionStatus:
        """Discard all pending state while keeping committed evidence."""
        logger.debug(
            "rolling back decoding transaction for 0x%x (call 0x%x) after attempt %d: %s",
            self.fva,
            self.decoded_at_va,
            self.attempts,
            reason or "unknown reason",
        )
        self._pending_delta_bytes = []
        self._pending_evidence = set()
        self.status = TransactionStatus.COMMITTED if self._committed else TransactionStatus.ROLLED_BACK
        return self.status

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
    @property
    def committed_evidence(self) -> Set[EvidenceKey]:
        return set(self._committed)

    @property
    def committed_delta_bytes(self) -> List[DeltaBytes]:
        return list(self._committed_delta_bytes)

    @property
    def succeeded(self) -> bool:
        return self.status == TransactionStatus.COMMITTED


class TransactionError(Exception):
    """Base class for decoding transaction errors."""


class TransactionStateError(TransactionError):
    """Raised when a lifecycle method is called in the wrong status."""


class TransactionExhausted(TransactionError):
    """Raised when a transaction is asked to retry beyond its retry budget."""

    def __init__(self, fva: int, decoded_at_va: int, attempts: int) -> None:
        super().__init__(
            f"decoding transaction for 0x{fva:x} (call 0x{decoded_at_va:x}) " f"exhausted after {attempts} attempt(s)"
        )
        self.fva = fva
        self.decoded_at_va = decoded_at_va
        self.attempts = attempts
