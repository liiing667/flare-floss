# Copyright 2026 Google LLC
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

"""Recursive layout tree nodes for binary structure."""

from __future__ import annotations

import abc
import bisect
import itertools
from typing import Any, Set, Dict, List, Tuple, Callable, Iterable, Optional, Sequence
from collections import defaultdict

import pefile
from pydantic import Field, BaseModel, ConfigDict

from floss.ranges import Range, Slice, OffsetRanges
from floss.tags.engine import check_is_xor, check_is_code, check_is_reloc
from floss.layout.types import Tag, TaggedString, ExtractedString

Tagger = Callable[[ExtractedString], Sequence[Tag]]

# process-wide counter that gives every Layout node a stable identity shared by
# range acceptance, string ownership, nesting generation, and report evidence.
_LAYOUT_ID_COUNTER = itertools.count()


class Structure(BaseModel):
    slice: Slice
    name: str


class Layout(BaseModel, abc.ABC):
    """
    recursively describe a region of a data, as a tree.
    the compute_layout routines construct this tree.

    each node in the tree (Layout), describes a range of the data.
    it may have children, which describes sub-ranges of the data.
    children don't overlap nor extend before/beyond the parent range.
    children are ordered by their offset in the data.
    children don't have to be contiguous - there can be gaps, or none at all.
    there are routines for traversing to the prior/next sibling, if any,
    and accessor properties for the parent and children.

    each node has a nice human readable name.
    each node has a list of strings that are assigned to it;
    a string is assigned to the deepest descendant node whose range contains
    the string's start offset, stopping at buffer domain boundaries.

    note that `Layout` is the abstract base class for nodes in the tree.
    subclasses are used to represent different types of regions,
    such as a PE file, a section, a segment, or a resource.
    subclasses can provide more specific behavior when it comes to tagging strings.
    """

    slice: Slice

    # human readable name
    name: str

    # stable identity of this node, shared across range acceptance, string
    # ownership, nesting generation, and the serializable report evidence.
    # a new id is minted for every node that is constructed.
    node_id: int = Field(default_factory=lambda: next(_LAYOUT_ID_COUNTER), init=False)

    # nesting generation: 0 for a layout root, incremented for every tree level.
    # set centrally by add_child so depth bookkeeping cannot drift between
    # PE/ELF/Mach-O/resource construction paths.
    generation: int = Field(default=0, init=False)

    parent: Optional["Layout"] = Field(default=None, init=False)

    # ordered by address
    # non-overlapping
    # may not cover the entire range (non-contiguous)
    children: Sequence["Layout"] = Field(default_factory=list, init=False)

    # this is populated by the call to extract_strings.
    # only strings not contained by the children are in this list.
    # so they come from before/between/after the children ranges.
    strings: List[TaggedString] = Field(default_factory=list, init=False)

    @property
    def predecessors(self) -> Iterable["Layout"]:
        """traverse to the prior siblings`"""
        if self.parent is None:
            return

        index = self.parent.children.index(self)
        if index == 0:
            return

        for i in range(index - 1, -1, -1):
            yield self.parent.children[i]

    @property
    def predecessor(self) -> Optional["Layout"]:
        """traverse to the prior sibling"""
        return next(iter(self.predecessors), None)

    @property
    def successors(self) -> Iterable["Layout"]:
        """traverse to the next siblings"""
        if self.parent is None:
            return

        index = self.parent.children.index(self)
        if index == len(self.parent.children) - 1:
            return

        for i in range(index + 1, len(self.parent.children)):
            yield self.parent.children[i]

    @property
    def successor(self) -> Optional["Layout"]:
        """traverse to the next sibling"""
        return next(iter(self.successors), None)

    def add_child(self, child: "Layout"):
        # this works in py3.11, though mypy gets confused,
        # maybe due to the use of the key function.
        bisect.insort(self.children, child, key=lambda c: c.slice.range.offset)  # type: ignore
        child.parent = self
        child.generation = self.generation + 1

    @property
    def offset(self) -> int:
        "convenience"
        return self.slice.range.offset

    @property
    def end(self) -> int:
        "convenience"
        return self.slice.range.end

    @property
    def depth(self) -> int:
        "number of edges from this node to the layout root (0 for a root)"
        return self.generation

    @property
    def buffer_domain_root(self) -> "Layout":
        """the root of this node's buffer domain.

        a buffer domain starts at the root node or at a node whose bytes live
        in a different buffer than its parent (such as an XOR-decoded nested
        layout). strings are extracted once per domain root.
        """
        node = self
        while node.parent is not None and node.parent.slice.buf is node.slice.buf:
            node = node.parent
        return node

    @property
    def domain_id(self) -> int:
        "identity of the buffer-domain root, used to namespace file offsets"
        return self.buffer_domain_root.node_id

    def _deepest(self, offset: int, end: int, buf: bytes, *, require_full: bool) -> "Layout":
        """shared descent rule for range acceptance and string ownership.

        at each level pick the most specific same-buffer sibling whose range
        starts at or before ``offset``: the greatest start offset wins among
        overlapping siblings, then the largest extent, then the newest node.
        when ``require_full`` is set, a candidate must fully contain
        ``[offset, end)``; otherwise containing just ``offset`` is enough (so a
        string crossing a node boundary stays whole and belongs to the node
        where it starts). descent stops when no sibling qualifies, returning
        the current node.
        """
        node = self
        while True:
            candidates = [
                child
                for child in node.children
                if child.slice.buf is buf
                and child.offset <= offset
                and (offset < child.end)
                and ((not require_full) or (end <= child.end))
            ]
            if not candidates:
                return node

            node = max(candidates, key=lambda c: (c.offset, c.end, c.node_id))

    def deepest_container(self, offset: int, end: int, buf: Optional[bytes] = None) -> "Layout":
        """find the deepest same-buffer node whose range fully contains
        ``[offset, end)`` (range acceptance for attaching a child).

        returns this node when no descendant accepts the whole range, which is
        how malformed/overlapping inputs that do not fit cleanly stay attached
        to their constructing parent instead of crashing or landing in a
        partially overlapping neighbor.
        """
        if buf is None:
            buf = self.slice.buf

        return self._deepest(offset, end, buf, require_full=True)

    def deepest_owner(self, offset: int, buf: Optional[bytes] = None) -> "Layout":
        """find the deepest same-buffer node whose range contains ``offset``
        (string ownership by the string's start offset).

        a string crossing a node boundary stays whole and is owned by the node
        in which it starts; overlapping siblings resolve to the most specific
        (greatest start offset) match.
        """
        if buf is None:
            buf = self.slice.buf

        return self._deepest(offset, offset + 1, buf, require_full=False)

    def accept_child(self, child: "Layout") -> "Layout":
        """attach ``child`` to the deepest node whose range fully contains it
        and that shares its buffer.

        centralizes range acceptance so that nested resources, gap/overlay
        segments, and malformed overlapping sections all use the same layout
        identity rule as string ownership. returns the node the child was
        attached to.
        """
        if child.slice.buf is self.slice.buf:
            container = self.deepest_container(child.offset, child.end, child.slice.buf)
        else:
            # a different buffer (e.g. XOR-decoded) starts its own domain; it
            # cannot be "contained" by an offset match against another buffer,
            # so it stays directly under the node that constructed it.
            container = self
        container.add_child(child)
        return container

    def _distribute_strings(self, strings: Iterable[ExtractedString]) -> None:
        """
        assign each string to the deepest descendant whose range contains the
        string's start offset, within the same buffer domain. siblings are
        sorted and normally don't overlap; if they do (malformed input), the
        child with the greatest start offset at or before the string wins.
        a child over a different buffer (XOR-decoded nested layout) owns its
        own buffer domain and never receives strings from this domain, and
        strings are never dropped just because such a child overlaps them.
        """
        strings = list(strings)
        if not strings:
            return

        # only same-buffer children can own strings from this domain; children
        # in another buffer (XOR-decoded nested layouts) extract their own.
        same_buffer_children = [c for c in self.children if c.slice.buf is self.slice.buf]
        if not same_buffer_children:
            self.strings.extend(strings)  # type: ignore
            return

        child_offsets = [c.offset for c in same_buffer_children]
        child_strings: Dict[int, List[ExtractedString]] = defaultdict(list)

        for s in strings:
            offset = s.slice.range.offset
            # consider every same-buffer child starting at or before the
            # string's start offset. with non-overlapping siblings only one
            # qualifies; with overlapping siblings the greatest start offset
            # (the most specific nested match) wins.
            owner: Optional["Layout"] = None
            for i in range(bisect.bisect_right(child_offsets, offset) - 1, -1, -1):
                candidate = same_buffer_children[i]
                if offset < candidate.end:
                    owner = candidate
                    break
            if owner is not None:
                child_strings[owner.node_id].append(s)
            else:
                self.strings.append(s)  # type: ignore

        # recurse with the shared node identity so deeper nesting (a resource
        # inside a section, or overlapping descendants) resolves identically.
        for child in same_buffer_children:
            bucket = child_strings.pop(child.node_id, None)
            if bucket:
                child._distribute_strings(bucket)

    def extract_strings(self, min_len: int) -> None:
        """
        find the strings in this layout and its children, recursively.

        strings are extracted once over the whole slice of each buffer domain
        root (the root node, or a node whose buffer differs from its parent's)
        and then distributed to the deepest node containing their start offset,
        so strings crossing node boundaries stay whole. this method must run
        before ``tag_strings``.
        """
        # imported here to avoid a circular import with floss.layout.extract
        from floss.layout.extract import extract_strings as extract_slice_strings

        is_buffer_domain_root = self.buffer_domain_root is self

        if is_buffer_domain_root:
            all_strings = extract_slice_strings(self.slice, min_len)
            self._distribute_strings(all_strings)

        for child in self.children:
            child.extract_strings(min_len)

    def tag_strings(self, taggers: Sequence[Tagger]):
        """
        tag the strings in this layout and its children, recursively.
        this means that the .strings field will contain TaggedStrings now
        (it used to contain ExtractedStrings).

        this can be overridden, if a subclass has more ways of tagging strings,
        such as a PE file and code/reloc regions.
        """
        string_counts: Dict[str, int] = defaultdict(int)

        tagged_strings: List[TaggedString] = []

        for string in self.strings:
            # at this moment, the list of strings contains only ExtractedStrings.
            # this routine will transform them into TaggedStrings.
            assert isinstance(string, ExtractedString)
            tags: Set[Tag] = set()

            string_counts[string.string] += 1

            if string_counts[string.string] > 1:
                tags.add("#duplicate")

            for tagger in taggers:
                tags.update(tagger(string))

            tagged_strings.append(TaggedString(string=string, tags=tags))
        self.strings = tagged_strings

        for child in self.children:
            child.tag_strings(taggers)

    def mark_structures(self, structures: Optional[Tuple[Dict[int, Structure], ...]] = (), **kwargs):
        """
        mark the structures that might be associated with each string, recursively.
        this means that the TaggedStrings may now have a non-empty .structure field.

        this can be overridden, if a subclass has a way of parsing structures,
        such as a PE file and all its data.
        """
        if structures:
            self._mark_string_structures(structures)

        for child in self.children:
            child.mark_structures(structures=structures, **kwargs)

    def _mark_string_structures(self, structures) -> None:
        """attach the first matching structure name to this node's own strings."""
        for string in self.strings:
            for structures_by_address in structures:
                structure = structures_by_address.get(string.offset)
                if structure:
                    string.structure = structure.name
                    break

    def _xor_reloc_code_taggers(self, xor_key, code_offsets, reloc_offsets) -> Tuple[Tagger, ...]:
        """the XOR, relocation, and code taggers common to the binary layouts."""

        def check_is_xor_tagger(s: ExtractedString) -> Sequence[Tag]:
            return check_is_xor(xor_key)

        def check_is_reloc_tagger(s: ExtractedString) -> Sequence[Tag]:
            return check_is_reloc(reloc_offsets, s)

        def check_is_code_tagger(s: ExtractedString) -> Sequence[Tag]:
            return check_is_code(code_offsets, s)

        return (check_is_xor_tagger, check_is_reloc_tagger, check_is_code_tagger)

    def _mark_structures_with(self, structures, structures_by_address, **kwargs) -> None:
        """mark structures on this node and recurse, threading ``structures_by_address``
        through the Section/Segment children (the layout types that represent
        binary sections)."""
        self._mark_string_structures((structures or ()) + (structures_by_address,))
        for child in self.children:
            if isinstance(child, (SectionLayout, SegmentLayout)):
                child.mark_structures(structures=(structures or ()) + (structures_by_address,), **kwargs)
            else:
                child.mark_structures(structures=structures, **kwargs)


class SectionLayout(Layout):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    section: Optional[pefile.SectionStructure] = None


class SegmentLayout(Layout):
    """region not covered by any section, such as PE header or overlay"""

    pass


class PELayout(Layout):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # xor key if the file was xor decoded
    xor_key: Optional[int]

    # file offsets of bytes that are part of the relocation table
    reloc_offsets: OffsetRanges

    # file offsets of bytes that are recognized as code
    code_offsets: OffsetRanges

    structures_by_address: Dict[int, Structure]

    def tag_strings(self, taggers: Sequence[Tagger]):
        super().tag_strings(
            tuple(taggers) + self._xor_reloc_code_taggers(self.xor_key, self.code_offsets, self.reloc_offsets)
        )

    def mark_structures(self, structures=(), **kwargs):
        self._mark_structures_with(structures, self.structures_by_address, **kwargs)


class ELFLayout(Layout):
    xor_key: Optional[int]

    # file offsets of bytes that are part of relocation sections
    relocation_offsets: OffsetRanges

    # file offsets of bytes that are recognized as code
    code_offsets: OffsetRanges

    structures_by_address: Dict[int, Structure]

    def tag_strings(self, taggers: Sequence[Tagger]):
        super().tag_strings(
            tuple(taggers) + self._xor_reloc_code_taggers(self.xor_key, self.code_offsets, self.relocation_offsets)
        )

    def mark_structures(self, structures: Optional[Tuple[Dict[int, Structure], ...]] = (), **kwargs):
        self._mark_structures_with(structures, self.structures_by_address, **kwargs)


class ResourceLayout(Layout):
    pass


class MachOLayout(Layout):
    arch: str
    structures_by_address: Dict[int, Structure] = Field(default_factory=dict)

    def mark_structures(self, structures=(), **kwargs):
        if self.structures_by_address:
            structures = structures + (self.structures_by_address,)
        super().mark_structures(structures=structures, **kwargs)

    def tag_strings(self, taggers: Sequence[Tagger]):
        super().tag_strings(taggers)


class MachOFatLayout(Layout):
    pass
