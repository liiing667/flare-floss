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

"""Project layout analysis onto FLOSS static/language string results."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from floss.results import ResultLayout, ResultString, StaticString, StringEncoding


def layout_encoding_to_string_encoding(encoding: str) -> StringEncoding:
    # layout ExtractedString.encoding is only "ascii" | "unicode"
    if encoding == "unicode":
        return StringEncoding.UTF16LE
    return StringEncoding.ASCII


def _walk_offset_index(
    layout: ResultLayout,
    index: Dict[Tuple[int, int], Tuple[ResultString, str]],
) -> None:
    for s in layout.strings:
        # namespace offsets by buffer-domain identity so an XOR-decoded nested
        # buffer that reuses file offsets cannot overwrite outer-buffer evidence.
        index[(s.domain_id, s.offset)] = (s, layout.name)
    for child in layout.children:
        _walk_offset_index(child, index)


def build_offset_index(layout: ResultLayout) -> Dict[Tuple[int, int], Tuple[ResultString, str]]:
    """Map (buffer-domain id, file offset) → (ResultString, section name)."""
    index: Dict[Tuple[int, int], Tuple[ResultString, str]] = {}
    _walk_offset_index(layout, index)
    return index


def static_strings_from_layout(layout: ResultLayout) -> List[StaticString]:
    """Flatten a serializable layout tree into enriched StaticString values."""
    index = build_offset_index(layout)
    # preserve every piece of evidence (including strings from different buffer
    # domains that happen to share an offset), ordered by domain then offset.
    items = sorted(index.items(), key=lambda kv: kv[0])
    out: List[StaticString] = []
    for (_domain_id, offset), (rs, section) in items:
        out.append(
            StaticString(
                string=rs.string,
                offset=offset,
                encoding=layout_encoding_to_string_encoding(rs.encoding),
                tags=list(rs.tags),
                section=section,
                structure=rs.structure or "",
            )
        )
    return out


def enrich_static_string(
    s: StaticString,
    offset_index: Dict[Tuple[int, int], Tuple[ResultString, str]],
    domain_id: int,
) -> StaticString:
    """Copy tags/section/structure from layout for a string with a file offset."""
    hit = offset_index.get((domain_id, s.offset))
    if not hit:
        return s
    rs, section = hit
    return StaticString(
        string=s.string,
        offset=s.offset,
        encoding=s.encoding,
        tags=list(rs.tags),
        section=section,
        structure=rs.structure or "",
    )


def enrich_static_strings(
    strings: List[StaticString],
    layout: Optional[ResultLayout] = None,
    *,
    offset_index: Optional[Dict[Tuple[int, int], Tuple[ResultString, str]]] = None,
    domain_id: Optional[int] = None,
) -> List[StaticString]:
    """
    Project layout tags/section/structure onto static strings by file offset.

    Pass a prebuilt ``offset_index`` when enriching multiple string lists so the
    layout tree is walked only once.

    Classic/language strings are read from the raw (outer) file buffer, so they
    are enriched against the root buffer domain (``layout.domain_id``); strings
    inside an XOR-decoded nested buffer are only reachable through that nested
    domain's own layout node and must not be matched by raw-buffer offsets.
    """
    if offset_index is None:
        if layout is None:
            return strings
        offset_index = build_offset_index(layout)
    if domain_id is None:
        if layout is None:
            return strings
        domain_id = layout.domain_id
    return [enrich_static_string(s, offset_index, domain_id) for s in strings]
