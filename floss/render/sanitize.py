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


"""Human-readable, terminal-safe rendering of untrusted strings.

Strings extracted from malware are attacker-controlled.  They may contain ANSI
escape sequences (which can move the cursor, change colors, or otherwise drive
the terminal), other control characters, or HTML fragments.  The *raw* value is
evidence and must be preserved in the JSON document, but every human-facing
renderer must show the same readable, single-line escaped form.

This module is the single source of truth for that form.  The text renderer and
the HTML viewer (``viewer/src/App.tsx``) implement the same rules so a value is
identified consistently across reports.
"""

from __future__ import annotations

# escapes that are both conventional and more readable than ``\uNNNN``
_NAMED_ESCAPES = {
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _escape_char(c: str) -> str:
    """return the readable escape for one character, or the character itself."""
    escaped = _NAMED_ESCAPES.get(c)
    if escaped is not None:
        return escaped

    # printable characters (including ordinary Unicode such as "é", "日", and
    # quotes) are left verbatim.  only invisible/control characters are escaped.
    if c.isprintable():
        return c

    # use JSON-style Unicode escapes so the readable form matches the escaped
    # characters in the JSON report (for example, ESC is ``\u001b`` in both).
    codepoint = ord(c)
    if codepoint <= 0xFFFF:
        return "\\u%04x" % codepoint
    return "\\U%08x" % codepoint


def sanitize(s: str, is_ascii_only: bool = False) -> str:
    """return a readable, lossless-enough rendering of ``s`` for humans.

    - control characters are escaped (``\\u001b``, ``\\u2028``, ...), so they
      cannot drive a terminal or disappear silently
    - backslashes are escaped so an original ``\\x1b`` remains distinguishable
      from an actual ESC character
    - printable Unicode and legal quote characters are preserved as-is
    - the result never contains a newline, so callers can keep one string per
      report line

    ``is_ascii_only`` is retained for backwards compatibility.  Printable
    non-ASCII characters are always preserved because dropping them would make
    the terminal, JSON, and HTML reports disagree about the evidence.
    """
    return "".join(_escape_char(c) for c in s)
