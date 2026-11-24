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

"""Regression tests for the shared layout identity.

A single layout identity (node id + nesting generation + buffer-domain id)
must drive range acceptance, string ownership, nesting generation, and the
serializable report evidence. The fixed fixture below deliberately combines
ordinary non-overlapping ranges, partially overlapping ranges, and a nested
child, plus an XOR-style child that switches the underlying buffer.
"""

from floss.ranges import Range, Slice
from floss.layout.base import SegmentLayout
from floss.layout.types import ExtractedString
from floss.layout.extract import collect_strings, extract_strings
from floss.results import ResultLayout, StaticString, StringEncoding
from floss.enrich import build_offset_index, enrich_static_strings, static_strings_from_layout


def _strings_on(node):
    return {s.string for s in node.strings}


def _all_strings(node):
    out = set()

    def walk(n):
        for s in n.strings:
            if s.slice.buf is node.slice.buf:
                out.add(s.string)
        for child in n.children:
            walk(child)

    walk(node)
    return out


def _make_combined_layout():
    """ordinary + overlapping + nested ranges in one buffer, plus a
    different-buffer (XOR-decoded style) nested domain."""
    buf = bytearray(0x400)
    # ordinary sibling strings
    buf[0x010:0x019] = b"NORMAL_1!"
    buf[0x210:0x219] = b"NORMAL_2!"
    # string in the overlap between c1 and the overlapping sibling `ov`
    buf[0x090:0x099] = b"OVERLAP!!"
    # string only covered by c1
    buf[0x180:0x189] = b"ONLY_C1!!"
    # string inside the nested child
    buf[0x055:0x05E] = b"DEEPSTR!!"
    # string in the raw outer buffer at an offset reused by the XOR domain
    buf[0x110:0x11B] = b"OUTER_STR!"

    root = SegmentLayout(name="root", slice=Slice.from_bytes(bytes(buf)))
    c1 = SegmentLayout(name="c1", slice=root.slice.slice(0x000, 0x200))
    c2 = SegmentLayout(name="c2", slice=root.slice.slice(0x200, 0x100))
    root.add_child(c1)
    root.add_child(c2)

    # partially overlaps the tail of c1 ([0x80, 0x100))
    ov = SegmentLayout(name="ov", slice=root.slice.slice(0x080, 0x080))
    root.add_child(ov)

    # fully nested inside c1
    deep = SegmentLayout(name="deep", slice=root.slice.slice(0x050, 0x020))
    c1.add_child(deep)

    # a different buffer reusing absolute offsets, like an XOR-decoded resource
    decoded = bytearray(0x100)
    decoded[0x10:0x1B] = b"INNER_STR!"
    xor_slice = Slice(buf=bytes(decoded), range=Range(offset=0x100, length=0x100), base_offset=0x100)
    xor_child = SegmentLayout(name="xor-nested", slice=xor_slice)
    root.add_child(xor_child)

    return root, c1, c2, ov, deep, xor_child


def test_node_identity_is_unique_and_stable():
    root, c1, c2, ov, deep, xor_child = _make_combined_layout()
    nodes = [root, c1, c2, ov, deep, xor_child]
    assert len({n.node_id for n in nodes}) == len(nodes)

    doc = ResultLayout.from_layout(root)
    assert doc.node_id == root.node_id
    doc_ids = set()

    def collect(node):
        doc_ids.add(node.node_id)
        for child in node.children:
            collect(child)

    collect(doc)
    assert doc_ids == {n.node_id for n in nodes}


def test_nesting_generation_tracks_tree_depth():
    root, c1, c2, ov, deep, xor_child = _make_combined_layout()
    assert root.generation == 0
    assert c1.generation == c2.generation == ov.generation == xor_child.generation == 1
    assert deep.generation == 2
    assert deep.depth == 2


def test_string_ownership_for_ordinary_overlapping_and_nested_ranges():
    root, c1, c2, ov, deep, xor_child = _make_combined_layout()
    root.extract_strings(4)

    assert "NORMAL_1!" in _strings_on(c1)
    assert "NORMAL_2!" in _strings_on(c2)
    # most specific (greatest start offset) overlapping sibling wins
    assert "OVERLAP!!" in _strings_on(ov)
    assert "OVERLAP!!" not in _strings_on(c1)
    # non-overlapping sub-interval still belongs to its section
    assert "ONLY_C1!!" in _strings_on(c1)
    # deepest descendant owns the nested string
    assert "DEEPSTR!!" in _strings_on(deep)

    # no same-buffer string is lost compared with a whole-file extraction of
    # the outer buffer; the XOR child contributes its own decoded domain and is
    # excluded from the outer-buffer comparison.
    got = sorted(
        (s.slice.range.offset, s.string)
        for s in collect_strings(root)
        if s.slice.buf is root.slice.buf
    )
    want = sorted((s.slice.range.offset, s.string) for s in extract_strings(root.slice, 4))
    assert got == want


def test_buffer_switch_keeps_each_domain_evidence():
    root, c1, c2, ov, deep, xor_child = _make_combined_layout()
    root.extract_strings(4)

    # the XOR domain extracts its own decoded strings ...
    assert "INNER_STR!" in _strings_on(xor_child)
    # ... and the outer-buffer string at a colliding absolute offset is kept
    # in the outer domain (here inside section c1), not dropped or reassigned
    assert "OUTER_STR!" in _all_strings(root)
    assert "OUTER_STR!" not in _strings_on(xor_child)

    # the two domains have distinct identities even though offsets overlap
    assert xor_child.domain_id == xor_child.node_id
    assert root.domain_id == root.node_id
    assert xor_child.domain_id != root.domain_id

    doc = ResultLayout.from_layout(root)
    index = build_offset_index(doc)
    assert index[(root.node_id, 0x110)][0].string == "OUTER_STR!"
    assert index[(xor_child.node_id, 0x110)][0].string == "INNER_STR!"

    # flattening preserves both pieces of evidence (no offset-key collision)
    statics = static_strings_from_layout(doc)
    assert {"OUTER_STR!", "INNER_STR!"}.issubset({s.string for s in statics})

    # raw-buffer enrichment only resolves against the root domain
    enriched = enrich_static_strings(
        [StaticString(string="OUTER_STR!", offset=0x110, encoding=StringEncoding.ASCII)],
        layout=doc,
    )
    # raw-buffer offsets resolve to outer-domain nodes only, never the XOR child
    assert enriched[0].section == "c1"
    assert enriched[0].string == "OUTER_STR!"


def test_report_evidence_carries_layout_identity():
    root, c1, c2, ov, deep, xor_child = _make_combined_layout()
    root.extract_strings(4)
    doc = ResultLayout.from_layout(root)

    def check(node):
        assert node.domain_id != -1 and node.node_id != -1
        for s in node.strings:
            assert s.node_id == node.node_id
            assert s.domain_id == node.domain_id
        for child in node.children:
            check(child)

    check(doc)


def test_accept_child_uses_shared_identity():
    root, c1, c2, ov, deep, xor_child = _make_combined_layout()

    # fully contained range attaches to the deepest same-buffer node
    nested = SegmentLayout(name="accepted-deep", slice=root.slice.slice(0x60, 0x10))
    container = root.accept_child(nested)
    assert container is deep
    assert nested.parent is deep and nested.generation == deep.generation + 1

    # a range that no child fully contains stays with the root instead of
    # crashing on or mis-assigning to a partially overlapping neighbor
    floater = SegmentLayout(name="floater", slice=root.slice.slice(0x1F0, 0x40))
    container = root.accept_child(floater)
    assert container is root


def test_cross_boundary_string_stays_whole_with_shared_identity():
    buf = bytearray(0x300)
    buf[0xF8:0x104] = b"CROSSING_STR"
    root = SegmentLayout(name="root", slice=Slice.from_bytes(bytes(buf)))
    c1 = SegmentLayout(name="c1", slice=root.slice.slice(0x000, 0x100))
    c2 = SegmentLayout(name="c2", slice=root.slice.slice(0x100, 0x100))
    root.add_child(c1)
    root.add_child(c2)

    root.extract_strings(4)

    whole = [s for s in collect_strings(root) if s.string == "CROSSING_STR"]
    assert len(whole) == 1
    assert isinstance(whole[0], ExtractedString)
    assert whole[0].slice.range.offset == 0xF8
    assert "CROSSING_STR" in _strings_on(c1)
