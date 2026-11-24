# Copyright 2022 Google LLC
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


import json
import datetime
import dataclasses

from floss.results import ResultDocument


class FlossJSONEncoder(json.JSONEncoder):
    """
    serializes FLOSS data structures into JSON.
    specifically:
      - dataclasses into their dict representation
      - datetimes to ISO8601 strings
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore [arg-type]
        if isinstance(o, datetime.datetime):
            if o.tzinfo is not None:
                o = o.astimezone(datetime.timezone.utc)
                return o.isoformat("T").replace("+00:00", "Z")
            return o.isoformat("T") + "Z"
        return super().default(o)


def sort_nested(obj):
    """recursively sort dict keys, preserving the given top-level ordering.

    the top level of a results document is kept in a fixed field order so the
    small ``metadata`` block always appears near the start of the file; nested
    dict keys are still emitted in sorted order.
    """
    if isinstance(obj, dict):
        return {key: sort_nested(value) for key, value in sorted(obj.items())}
    if isinstance(obj, list):
        return [sort_nested(value) for value in obj]
    return obj


# top-level key order of a results document.
# metadata is deliberately first: it is small, so a results document is always
# recognizable from its leading bytes by detect_file_type.
TOP_LEVEL_KEYS = ("metadata", "analysis", "strings", "layout")


def _escape_nonprintable_json(text: str) -> str:
    """escape non-ASCII control/invisible characters that JSON permits verbatim.

    ``json.dumps`` always escapes ASCII control characters, but it leaves C1
    controls (for example U+009B, the ANSI CSI introducer), DEL, bidi overrides,
    and Unicode line separators untouched when ``ensure_ascii=False``.  Those
    characters can drive a terminal or otherwise hide the evidence, so escape
    them while keeping ordinary printable Unicode human-readable.
    """
    rendered: list[str] = []
    for c in text:
        if c.isprintable():
            rendered.append(c)
            continue

        codepoint = ord(c)
        if codepoint <= 0xFFFF:
            rendered.append("\\u%04x" % codepoint)
        else:
            # JSON has no \U escape; use a surrogate pair
            codepoint -= 0x10000
            rendered.append("\\u%04x\\u%04x" % (0xD800 + (codepoint >> 10), 0xDC00 + (codepoint & 0x3FF)))
    return "".join(rendered)


def _render_dict(data: dict) -> str:
    """serialize a results-document dict with fixed top-level ordering."""
    # keep the fixed top-level ordering (metadata first so the document stays
    # detectable from its leading bytes), then append any future/extra keys so
    # they are never silently dropped
    extra_keys = [key for key in data if key not in TOP_LEVEL_KEYS]
    ordered_keys = list(TOP_LEVEL_KEYS) + extra_keys
    top = {key: sort_nested(data[key]) for key in ordered_keys if key in data}
    # keep ordinary Unicode verbatim so the JSON report shows the same evidence
    # as the terminal and HTML reports.  json.dumps still escapes control
    # characters (for example ESC becomes \u001b) and JSON metacharacters.
    rendered = json.dumps(top, cls=FlossJSONEncoder, sort_keys=False, ensure_ascii=False)
    return _escape_nonprintable_json(rendered)


def render(doc: ResultDocument) -> str:
    return _render_dict(dataclasses.asdict(doc))
