# Copyright 2017 Google LLC
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
import envi.memory
from envi.archs.i386 import emu as i386emu

import floss.decoding_manager as decoding_manager
from floss.api_hooks import MemoryAllocationHook
from floss.decoding_manager import (
    Delta,
    Snapshot,
    MapsTooLargeError,
    EmulationGeneration,
    get_map_size,
    make_snapshot,
    add_memory_map,
    emulation_generation,
    get_emulation_generation,
)


@pytest.fixture
def emu():
    emulator = i386emu.IntelEmulator()
    emulator.addMemoryMap(0x10000, envi.memory.MM_RWX, "base", b"\x00" * 0x1000)
    return emulator


def test_get_map_size_sums_all_maps(emu):
    # regression: the size used to be overwritten with each iteration,
    # so only the size of the last map was returned
    emu.addMemoryMap(0x20000, envi.memory.MM_RWX, "a", b"\x00" * 0x1000)
    emu.addMemoryMap(0x30000, envi.memory.MM_RWX, "b", b"\x00" * 0x2000)
    emu.addMemoryMap(0x40000, envi.memory.MM_RWX, "c", b"\x00" * 0x3000)

    assert get_map_size(emu) == 0x1000 + 0x1000 + 0x2000 + 0x3000


def test_many_small_maps_share_one_budget(emu):
    # many small maps must accumulate against the same budget instead of
    # each one being measured against only the previous map size
    budget = 0x4000
    with emulation_generation(emu, max_maps_size=budget) as generation:
        for _ in range(3):
            emu.addMemoryMap(
                generation.emu.getMemoryMaps()[-1][0] + 0x10000,
                envi.memory.MM_RWX,
                "small",
                b"\x00" * 0x1000,
            )

        assert generation.total_maps_size == 0x1000 + 3 * 0x1000

        with pytest.raises(MapsTooLargeError):
            emu.addMemoryMap(0x90000, envi.memory.MM_RWX, "over", b"\x00" * 0x1000)

        # the rejected map must not have been installed in the emulator
        assert get_map_size(emu) == 0x1000 + 3 * 0x1000


def test_oversized_map_is_rejected_without_changing_state(emu):
    with emulation_generation(emu, max_maps_size=0x2000) as generation:
        maps_before = list(emu.getMemoryMaps())

        with pytest.raises(MapsTooLargeError):
            emu.addMemoryMap(0x50000, envi.memory.MM_RWX, "huge", b"\x00" * 0x10000)

        assert emu.getMemoryMaps() == maps_before
        assert generation.total_maps_size == 0x1000


def test_budget_boundary_is_inclusive(emu):
    budget = 0x2000
    with emulation_generation(emu, max_maps_size=budget) as generation:
        # one map that lands exactly on the budget is still accepted
        emu.addMemoryMap(0x20000, envi.memory.MM_RWX, "edge", b"\x00" * 0x1000)
        assert generation.total_maps_size == budget

        # a single additional byte over the budget is rejected
        with pytest.raises(MapsTooLargeError):
            emu.addMemoryMap(0x30000, envi.memory.MM_RWX, "byte", b"\x00" * 1)


def test_add_memory_map_helper_routes_through_generation(emu):
    with emulation_generation(emu, max_maps_size=0x2000):
        add_memory_map(emu, 0x20000, envi.memory.MM_RWX, "via-helper", b"\x00" * 0x1000)
        assert get_map_size(emu) == 0x2000

        with pytest.raises(MapsTooLargeError):
            add_memory_map(emu, 0x30000, envi.memory.MM_RWX, "via-helper", b"\x00" * 0x1000)


def test_add_memory_map_helper_without_generation_is_unguarded(emu):
    assert get_emulation_generation(emu) is None
    add_memory_map(emu, 0x20000, envi.memory.MM_RWX, "plain", b"\x00" * 0x10)
    assert get_map_size(emu) == 0x1010


def test_snapshot_is_published_by_the_active_generation(emu):
    with emulation_generation(emu) as generation:
        snapshot = make_snapshot(emu)

        assert isinstance(snapshot, Snapshot)
        assert snapshot.generation == generation.id
        assert snapshot.sp == emu.getStackCounter()
        assert snapshot.pc == emu.getProgramCounter()


def test_generation_ids_increase(emu):
    with emulation_generation(emu) as first:
        assert first.id == 1
    with emulation_generation(emu) as second:
        assert second.id == first.id + 1


def test_failed_snapshot_recovers_last_good_state(emu):
    with emulation_generation(emu, max_maps_size=0x2000) as generation:
        # establish a good, in-budget baseline
        good_snapshot = make_snapshot(emu)

        # simulate an oversized map installed outside of the admission check
        generation._original_add_memory_map(0x50000, envi.memory.MM_RWX, "rogue", b"\x00" * 0x5000)
        assert get_map_size(emu) > generation.max_maps_size

        with pytest.raises(MapsTooLargeError):
            make_snapshot(emu)

        # the rogue state must have been rolled back so it cannot pollute
        # the rest of this emulation context
        assert get_map_size(emu) == 0x1000
        assert generation.total_maps_size == 0x1000
        assert emu.getMemorySnap() == good_snapshot.memory

        # emulation can continue within the same generation afterwards
        emu.addMemoryMap(0x60000, envi.memory.MM_RWX, "recovered", b"\x00" * 0x800)
        assert generation.total_maps_size == 0x1800
        assert make_snapshot(emu).generation == generation.id


def test_generation_detaches_on_cleanup(emu):
    original_add_memory_map = emu.addMemoryMap
    with emulation_generation(emu) as generation:
        assert get_emulation_generation(emu) is generation
        assert emu.addMemoryMap != original_add_memory_map

    assert get_emulation_generation(emu) is None
    assert "addMemoryMap" not in emu.__dict__


def test_closed_generation_rejects_maps_and_snapshots(emu):
    with emulation_generation(emu, max_maps_size=0x2000) as generation:
        pass

    assert generation.is_active() is False
    with pytest.raises(MapsTooLargeError):
        generation.add_memory_map(0x20000, envi.memory.MM_RWX, "late", b"\x00" * 0x10)
    with pytest.raises(MapsTooLargeError):
        generation.make_snapshot()


def test_legacy_snapshot_fields_remain_available(emu):
    # snapshots taken without a generation keep the historical three fields
    snapshot = make_snapshot(emu)

    assert hasattr(snapshot, "memory")
    assert hasattr(snapshot, "sp")
    assert hasattr(snapshot, "pc")
    assert snapshot.generation == 0

    # the dataclass remains constructible with the original positional fields
    legacy = Snapshot(emu.getMemorySnap(), 1, 2)
    assert legacy.memory == snapshot.memory
    assert legacy.sp == 1
    assert legacy.pc == 2
    assert legacy.generation == 0

    delta = Delta(snapshot, legacy)
    assert delta.pre is snapshot
    assert delta.post is legacy


def test_legacy_make_snapshot_enforces_budget(monkeypatch, emu):
    assert get_emulation_generation(emu) is None
    monkeypatch.setattr(decoding_manager, "MAX_MAPS_SIZE", 0x1000)

    make_snapshot(emu)  # exactly at the limit is fine

    emu.addMemoryMap(0x20000, envi.memory.MM_RWX, "extra", b"\x00" * 1)
    with pytest.raises(MapsTooLargeError):
        make_snapshot(emu)


def test_allocation_hook_flood_is_stopped_by_shared_budget():
    # end-to-end check with the real MemoryAllocationHook: many small
    # allocations used to bypass the total budget because only the size of
    # the most recently added map was measured
    emulator = i386emu.IntelEmulator()
    emulator.addMemoryMap(0x10000, envi.memory.MM_RWX, "base", b"\x00" * 0x1000)
    emulator.addMemoryMap(0x70000, envi.memory.MM_RWX, "stack", b"\x00" * 0x10000)
    emulator.setStackCounter(0x70000 + 0x8000)
    emulator.writeMemory(emulator.getStackCounter(), (0x1234).to_bytes(4, "little"))

    hook = MemoryAllocationHook()
    api = (0, 0, "cdecl", "malloc", ())
    budget = 0x11000 + 0x3000

    with emulation_generation(emulator, max_maps_size=budget) as generation:
        # ordinary, in-budget allocations keep working
        assert hook(emulator, api, [0x100]) is True

        rejected = False
        for _ in range(1000):
            try:
                hook(emulator, api, [0x1000])
            except MapsTooLargeError:
                rejected = True
                break

        assert rejected
        # the bookkeeping and the live emulator agree and stay within budget
        assert generation.total_maps_size == get_map_size(emulator)
        assert generation.total_maps_size <= budget

        # snapshots can still be published after the failed allocation
        snapshot = make_snapshot(emulator)
        assert snapshot.generation == generation.id
