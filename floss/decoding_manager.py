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


import contextlib
from typing import List, Tuple, Optional
from dataclasses import dataclass

import envi.bits
import viv_utils
import envi.memory
import vivisect.const
import viv_utils.emulator_drivers
from envi import Emulator

import floss.utils
import floss.logging_

from . import api_hooks
from .const import DS_MAX_ADDRESS_REVISITS_EMULATION

logger = floss.logging_.getLogger(__name__)
MAX_MAPS_SIZE = 1024 * 1024 * 100  # 100MB max memory allocated in an emulator instance

# attribute under which the active EmulationGeneration is attached to an emulator
_EMULATION_GENERATION_ATTR = "_floss_emulation_generation"


def is_import(emu, va):
    """
    Return True if the given VA is that of an imported function.
    """
    t = emu.getVivTaint(va)
    if t is None:
        return False

    _, ttype, tinfo = t
    return ttype == "import" and isinstance(tinfo, tuple) and len(tinfo) >= 3 and tinfo[2] == vivisect.const.LOC_IMPORT


# type aliases for envi.memory map
MemoryMapDescriptor = Tuple[
    # va
    int,
    # size
    int,
    # perms
    int,
    # name
    str,
]

# type aliases for envi.memory map
MemoryMap = Tuple[
    # start
    int,
    # end
    int,
    # descriptor
    MemoryMapDescriptor,
    # content
    bytes,
]

# type aliases for envi.memory map
Memory = List[MemoryMap]


@dataclass
class Snapshot:
    """
    A snapshot of the state of the CPU and memory.

    Attributes:
        memory: a snapshot of the memory contents
        sp: the stack counter
        pc: the instruction pointer
    """

    memory: Memory
    sp: int
    pc: int
    # the emulation generation in which this snapshot was taken;
    # snapshots created outside of a generation (e.g. by callers that still
    # use the legacy helpers directly) keep the default value of zero
    generation: int = 0


def get_map_size(emu: Emulator) -> int:
    """
    Return the total size of all memory maps currently installed in the emulator.
    """
    return sum(mapsize for _, mapsize, _, _ in emu.getMemoryMaps())


class MapsTooLargeError(Exception):
    pass


class EmulationGeneration:
    """
    A single generation of emulator state that owns the mapped-memory budget.

    Map admission, the total-size budget, snapshot publication and cleanup
    recovery all reference the same generation, so that:

      - every accepted map (no matter how small) counts against one shared
        budget instead of only the size of the most recently added map;
      - a map that would exceed the budget is never installed in the emulator;
      - a snapshot is only published while the generation is still within
        budget and matches the live emulator state;
      - a failed publication restores the emulator to the last good state of
        this generation instead of leaking the oversized maps into the
        following context.
    """

    def __init__(self, emu: Emulator, max_maps_size: int = MAX_MAPS_SIZE):
        self.emu = emu
        self.max_maps_size = max_maps_size
        # monotonically increasing identifier, bumped for every generation
        self.id = getattr(emu, "_floss_emulation_generation_id", 0) + 1
        setattr(emu, "_floss_emulation_generation_id", self.id)
        # the authoritative, in-budget emulator memory state of this generation
        self._baseline_memory_snap = emu.getMemorySnap()
        self._total_maps_size = get_map_size(emu)
        self._closed = False
        self._original_add_memory_map = emu.addMemoryMap
        self._had_instance_add_memory_map = "addMemoryMap" in emu.__dict__

    @property
    def total_maps_size(self) -> int:
        return self._total_maps_size

    def is_active(self) -> bool:
        return not self._closed and get_emulation_generation(self.emu) is self

    def add_memory_map(self, va: int, perms: int, fname: str, bytez: bytes, align: Optional[int] = None) -> int:
        """
        Admit a new memory map into this generation if it fits the budget.

        Raises MapsTooLargeError (without touching the emulator state) when
        accepting the map would exceed the total mapped-memory budget.
        """
        if self._closed:
            raise MapsTooLargeError("emulation generation %d is closed" % self.id)

        map_size = len(bytez)
        if align:
            map_size = envi.bits.align(map_size, align)

        if self._total_maps_size + map_size > self.max_maps_size:
            logger.debug(
                "rejecting memory map of size 0x%x, emulator mapped too much memory: 0x%x",
                map_size,
                self._total_maps_size,
            )
            raise MapsTooLargeError()

        map_size = self._original_add_memory_map(va, perms, fname, bytez, align)
        self._total_maps_size += map_size
        return map_size

    def make_snapshot(self) -> Snapshot:
        """
        Publish a snapshot of the current CPU and memory of this generation.

        If the live emulator no longer matches the generation (for example
        because an oversized map was installed outside of the budget check),
        the emulator is recovered to the last good state of this generation
        and MapsTooLargeError is raised instead of publishing a stale snapshot.
        """
        if self._closed:
            raise MapsTooLargeError("emulation generation %d is closed" % self.id)

        current_size = get_map_size(self.emu)
        if current_size > self.max_maps_size or current_size != self._total_maps_size:
            logger.debug("emulator mapped too much memory: 0x%x, recovering generation %d", current_size, self.id)
            self.recover()
            raise MapsTooLargeError()

        snapshot = Snapshot(
            self.emu.getMemorySnap(),
            self.emu.getStackCounter(),
            self.emu.getProgramCounter(),
            self.id,
        )
        # the freshly published snapshot becomes the new last-good state
        self._baseline_memory_snap = snapshot.memory
        return snapshot

    def recover(self) -> None:
        """
        Restore the emulator memory to the last good state of this generation.

        Used to clean up oversized or otherwise invalid state so that failed
        snapshots cannot pollute subsequent emulation contexts.
        """
        if not self._closed:
            self.emu.setMemorySnap(self._baseline_memory_snap)
            self._total_maps_size = get_map_size(self.emu)

    def close(self) -> None:
        """
        Detach the generation from the emulator and restore the unguarded
        addMemoryMap implementation.
        """
        if self._closed:
            return
        self._closed = True
        if self._had_instance_add_memory_map:
            self.emu.addMemoryMap = self._original_add_memory_map
        else:
            del self.emu.addMemoryMap
        if get_emulation_generation(self.emu) is self:
            delattr(self.emu, _EMULATION_GENERATION_ATTR)

    @contextlib.contextmanager
    def guard_memory_maps(self):
        """
        Route emulator.addMemoryMap calls through this generation's budget
        check for the duration of the context.
        """
        self.emu.addMemoryMap = self.add_memory_map
        setattr(self.emu, _EMULATION_GENERATION_ATTR, self)
        try:
            yield self
        finally:
            self.close()


def get_emulation_generation(emu: Emulator) -> Optional["EmulationGeneration"]:
    """
    Return the EmulationGeneration currently guarding the emulator, or None.
    """
    return getattr(emu, _EMULATION_GENERATION_ATTR, None)


def add_memory_map(emu: Emulator, va: int, perms: int, fname: str, bytez: bytes, align: Optional[int] = None) -> int:
    """
    Add a memory map via the emulator's active generation when one exists,
    otherwise fall back to the unguarded emulator implementation.
    """
    generation = get_emulation_generation(emu)
    if generation is not None:
        return generation.add_memory_map(va, perms, fname, bytez, align)
    return emu.addMemoryMap(va, perms, fname, bytez, align)


def make_snapshot(emu: Emulator) -> Snapshot:
    """
    Create a snapshot of the current CPU and memory.

    When an EmulationGeneration is active, the snapshot is published through
    that generation (including budget enforcement and cleanup recovery);
    otherwise the legacy size check is used.
    """
    generation = get_emulation_generation(emu)
    if generation is not None:
        return generation.make_snapshot()

    if get_map_size(emu) > MAX_MAPS_SIZE:
        logger.debug("emulator mapped too much memory: 0x%x", get_map_size(emu))
        raise MapsTooLargeError()
    return Snapshot(emu.getMemorySnap(), emu.getStackCounter(), emu.getProgramCounter())


@contextlib.contextmanager
def emulation_generation(emu: Emulator, max_maps_size: int = MAX_MAPS_SIZE):
    """
    Run a block of emulator work inside a single EmulationGeneration.

    Map admission, the memory budget, snapshot publication and cleanup
    recovery all share the same generation for the duration of the block.
    """
    generation = EmulationGeneration(emu, max_maps_size=max_maps_size)
    with generation.guard_memory_maps():
        yield generation


@dataclass
class Delta:
    """
    a pair of snapshots from before and after an operation.
    facilitates diffing the state of an emulator.
    """

    pre: Snapshot
    post: Snapshot


class DeltaCollectorHook(viv_utils.emulator_drivers.Hook):
    """
    hook that collects Deltas at each imported API call.
    """

    def __init__(self, pre_snap: Snapshot):
        super().__init__()
        self._pre_snap = pre_snap
        self.deltas: List[Delta] = []

    def __call__(self, emu, api, argv):
        if is_import(emu, emu.getProgramCounter()):
            # TODO add apis to ignore here, e.g.
            #  "kernel32.GetSystemTime", "ntdll.RtlFreeHeap", "ntdll.RtlAllocateHeap",
            #  callname = driver._emu.getCallApi(driver._emu.getProgramCounter())[3]
            try:
                # TODO optimize - may leverage writelog
                #  reduce duplicate deltas
                #  reduce redundant (unchanged) data in each delta
                self.deltas.append(Delta(self._pre_snap, make_snapshot(emu)))
            except MapsTooLargeError:
                _, _, _, name, _ = api
                logger.debug("despite call to import %s, maps too large, not extracting strings", name)
                pass


def emulate_function(
    emu: Emulator, function_index, fva: int, return_address: int, max_instruction_count: int
) -> List[Delta]:
    """
    Emulate a function and collect snapshots at each interesting place.
    These interesting places include calls to imported API functions
     and the final state of the emulator.
    Emulation continues until the return address is hit, or
     the given max_instruction_count is hit.
    Some library functions are shimmed, such as memory allocation routines.
    This helps "normal" routines emulate correct using standard library function.
    These include:
      - GetProcessHeap
      - RtlAllocateHeap
      - AllocateHeap
      - malloc

    :type function_index: viv_utils.FunctionIndex
    :param fva: The start address of the function to emulate.
    :param return_address: The expected return address of the function.
     Emulation stops here.
    :param max_instruction_count: The max number of instructions to emulate.
     This helps avoid unexpected infinite loops.
    """
    # map admission, the size budget, snapshot publication and cleanup
    # recovery all share one emulation generation for this function run
    with emulation_generation(emu) as generation:
        try:
            pre_snap = make_snapshot(emu)
        except MapsTooLargeError:
            logger.warning("initial snapshot mapped too much memory, can't extract strings")
            return []

        delta_collector = DeltaCollectorHook(pre_snap)

        try:
            logger.debug("Emulating function at 0x%08x", fva)
            driver = viv_utils.emulator_drivers.DebuggerEmulatorDriver(
                emu, repmax=256, max_hit=DS_MAX_ADDRESS_REVISITS_EMULATION, max_insn=max_instruction_count
            )
            monitor = api_hooks.ApiMonitor(function_index)
            driver.add_monitor(monitor)
            driver.add_hook(delta_collector)

            with api_hooks.defaultHooks(driver):
                driver.run_to_va(return_address)

        except viv_utils.emulator_drivers.BreakpointHit as e:
            # TODO track/shortcut instances of this
            if e.reason == "max_insn":
                logger.debug("Halting as emulation has escaped!")
        except envi.InvalidInstruction as e:
            logger.debug("vivisect encountered an invalid instruction. will continue processing. %s", e)
        except envi.UnsupportedInstruction as e:
            logger.debug("vivisect encountered an unsupported instruction. will continue processing. %s", e)
        except envi.BreakpointHit as e:
            logger.debug("vivisect encountered an unexpected emulation breakpoint. will continue processing. %s", e)
        except envi.exc.SegmentationViolation as e:
            tos_val = floss.utils.get_stack_value(emu, 0)
            logger.debug("%s: top of stack (return address): 0x%x", e, tos_val)
        except envi.exc.DivideByZero as e:
            logger.debug("vivisect encountered an emulation error. will continue processing. %s", e)
        except viv_utils.emulator_drivers.StopEmulation:
            pass
        except Exception:
            # we cheat here a bit and skip over various errors, check this for improvements and debugging
            logger.debug("vivisect encountered an unexpected exception. will continue processing.", exc_info=True)
        logger.debug("Ended emulation at 0x%08x", emu.getProgramCounter())

        deltas = delta_collector.deltas

        try:
            deltas.append(Delta(pre_snap, make_snapshot(emu)))
        except MapsTooLargeError:
            logger.debug("failed to create final snapshot, emulator mapped too much memory, skipping")
            # recover the emulator to the last good state of this generation
            # so the failed snapshot cannot pollute subsequent contexts
            generation.recover()

        return deltas
