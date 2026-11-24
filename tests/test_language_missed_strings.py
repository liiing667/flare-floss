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

from floss.results import StaticString, StringEncoding
from floss.language.utils import (
    get_string_span,
    is_span_contained,
    get_missed_strings,
    strip_found_lang_string,
    cut_contained_lang_spans,
)

MIN_LEN = 4


def ss(string: str, offset: int, encoding: StringEncoding = StringEncoding.UTF8) -> StaticString:
    return StaticString(string=string, offset=offset, encoding=encoding)


def test_get_string_span_uses_absolute_file_offset_and_utf8_byte_length():
    # "e-acute" is two UTF-8 bytes, so the span is byte-based rather than char-based
    s = ss("ab\u00e9cd", 0x800)
    assert get_string_span(s) == (0x800, 0x806)


def test_exact_match_span_is_consumed():
    # identical absolute spans: the static string is fully accounted for
    static = ss("complete", 0x400)
    lang = ss("complete", 0x400)

    assert is_span_contained(lang, static)
    assert get_missed_strings([static], [lang], MIN_LEN) == []


def test_overlapping_span_keeps_surrounding_bytes_as_missed_evidence():
    # language string covers only a sub-span of the larger static string
    static = ss("prefix-middle-suffix-long", 0x300)
    lang = ss("middle", 0x307)

    assert is_span_contained(lang, static)

    trimmed, remaining = strip_found_lang_string(static, lang)
    assert trimmed == ss("prefix--suffix-long", 0x300)
    assert remaining == len("prefix--suffix-long")

    # cutting keeps each surviving piece at its own absolute file offset
    pieces, found = cut_contained_lang_spans(static, [lang])
    assert found is True
    assert pieces == [ss("prefix-", 0x300), ss("-suffix-long", 0x30D)]

    missed = get_missed_strings([static], [lang], MIN_LEN)
    assert missed == [ss("prefix-", 0x300), ss("-suffix-long", 0x30D)]


def test_prefix_less_trim_moves_offset_to_end_of_language_span():
    static = ss("middle-suffix", 0x307)
    lang = ss("middle", 0x307)

    trimmed, _ = strip_found_lang_string(static, lang)
    assert trimmed == ss("-suffix", 0x30D)


def test_misaligned_offset_does_not_strip_same_named_content_elsewhere():
    # same textual content exists in two distinct file regions
    region_a = ss("hello!", 0x100, StringEncoding.ASCII)
    region_b = ss("xxhello!yy", 0x200, StringEncoding.ASCII)

    # language entry exactly covering region A must leave region B untouched
    missed = get_missed_strings([region_a, region_b], [ss("hello!", 0x100)], MIN_LEN)
    assert missed == [region_b]

    # a language entry whose offset lands inside region B only strips region B;
    # region A remains missed evidence with its own absolute offset
    missed = get_missed_strings([region_a, region_b], [ss("hello!", 0x202)], MIN_LEN)
    assert region_a in missed
    assert all(m.offset in (0x100, 0x200, 0x208) for m in missed)


def test_language_string_near_but_outside_candidate_span_is_not_consumed():
    # the language offset is merely *near* the candidate window, not inside it
    static = ss("standalone", 0x500)
    lang = ss("standalone", 0x900)

    assert not is_span_contained(lang, static)
    assert get_missed_strings([static], [lang], MIN_LEN) == [static]


def test_strip_rejects_bytes_that_do_not_match_at_shared_offset():
    # same-named content at an offset that does not actually align with the
    # claimed bytes must raise instead of silently deleting wrong content
    static = ss("xxhello!yy", 0x200)
    lang = ss("hello!", 0x201)  # [0x201, 0x207) covers "xhello", not "hello!"

    with pytest.raises(ValueError, match="do not match"):
        strip_found_lang_string(static, lang)


def test_ordinary_ascii_results_are_preserved():
    ascii_statics = [
        ss("ascii text", 0x500, StringEncoding.ASCII),
        ss("other ascii", 0x600, StringEncoding.ASCII),
    ]

    missed = get_missed_strings(ascii_statics, [ss("unrelated", 0x700)], MIN_LEN)
    assert missed == ascii_statics


def test_multibyte_utf8_strip_keeps_absolute_offset():
    static = ss("ab\u00e9cd", 0x800)
    lang = ss("\u00e9", 0x802)  # spans [0x802, 0x804)

    trimmed, _ = strip_found_lang_string(static, lang)
    assert trimmed == ss("abcd", 0x800)


def test_each_language_string_is_consumed_only_once():
    # two language sub-spans inside one static string are both handled
    static = ss("aaaa-foo-bbb-foo-cccc", 0x1000)
    langs = [ss("foo", 0x1005), ss("foo", 0x100D)]

    missed = get_missed_strings([static], langs, MIN_LEN)
    assert missed == [ss("aaaa-", 0x1000), ss("-bbb-", 0x1008), ss("-cccc", 0x1010)]


def test_value_equal_language_strings_at_distinct_spans_are_both_handled():
    # identical StaticString values at different spans must not be conflated
    static = ss("foo-aaaa-foo", 0x2000)
    langs = [ss("foo", 0x2000), ss("foo", 0x2009)]

    missed = get_missed_strings([static], langs, MIN_LEN)
    assert missed == [ss("-aaaa-", 0x2003)]


def test_min_length_drops_short_surviving_pieces_but_keeps_longer_ones():
    # the short "xx" remnant is filtered while the long remnant is retained
    static = ss("xx-middle-remnant", 0x3000)
    lang = ss("middle", 0x3003)

    missed = get_missed_strings([static], [lang], MIN_LEN)
    # "xx-" is 3 chars (< MIN_LEN) and dropped; "-remnant" survives
    assert missed == [ss("-remnant", 0x3009)]
