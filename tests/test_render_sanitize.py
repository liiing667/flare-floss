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

"""Consistent rendering of attacker-controlled strings across report formats."""

import json

import floss.render.html
import floss.render.json
import floss.render.default
from floss.results import (
    Strings,
    Analysis,
    Metadata,
    ResultLayout,
    ResultString,
    ResultDocument,
)
from floss.render.sanitize import sanitize

MALICIOUS = "\x1b[31m<script>alert('x')</script>\x1b[0m café 日本語 \"quoted\" \\x1b \u009b \u202e"


def _doc() -> ResultDocument:
    layout = ResultLayout(
        name="pe",
        offset=0,
        length=0x1000,
        strings=[ResultString(string=MALICIOUS, offset=0x100, size=len(MALICIOUS), encoding="ASCII")],
    )
    return ResultDocument(
        metadata=Metadata(file_path="sample.exe", min_length=4),
        analysis=Analysis(enable_stack_strings=False, enable_tight_strings=False, enable_decoded_strings=False),
        strings=Strings(static_strings=[]),
        layout=layout,
    )


def test_sanitize_escapes_controls_but_keeps_printable_text():
    rendered = sanitize(MALICIOUS)
    assert "\x1b" not in rendered
    assert "\u009b" not in rendered
    assert "\u202e" not in rendered
    assert "\\u001b[31m" in rendered
    assert "\\u009b" in rendered
    assert "\\u202e" in rendered
    # backslashes are escaped so a literal "\u001b" is distinguishable from ESC
    assert "\\\\x1b" in rendered
    assert "café" in rendered
    assert "日本語" in rendered
    assert '"quoted"' in rendered


def test_text_report_is_terminal_safe_and_readable():
    out = floss.render.default.render(_doc(), False, False, "never")
    assert not any(c in out for c in "\x1b\u009b\u202e")
    assert "\\u001b[31m" in out
    assert "café 日本語" in out
    assert "<script>" in out


def test_json_report_preserves_raw_value_and_printable_unicode():
    raw = floss.render.json.render(_doc())
    assert not any(c in raw for c in "\x1b\u009b\u202e")
    assert "café" in raw
    assert '"quoted"' not in raw  # JSON syntax escapes the quotes

    loaded = json.loads(raw)
    assert loaded["layout"]["strings"][0]["string"] == MALICIOUS


def test_html_report_escapes_script_and_keeps_raw_value_in_document():
    template = "<script>window.flossResults = null; /* FLOSS_RESULTS */</script>"
    html = floss.render.html.render(_doc(), template=template)
    assigned = html.split("window.flossResults = ", 1)[1].split("; /* FLOSS_RESULTS */", 1)[0]

    assert not any(c in assigned for c in "\x1b\u009b\u202e")
    assert "\\u003cscript\\u003e" in assigned
    assert "café" in assigned
    assert json.loads(assigned)["layout"]["strings"][0]["string"] == MALICIOUS
