import re
import array
import struct
import hashlib
from typing import Set, List, Tuple, Iterable, Optional
from dataclasses import dataclass

import pefile
import tabulate
from typing_extensions import TypeAlias

import floss.utils
from floss.results import StaticString, StringEncoding
from floss.render.sanitize import sanitize

VA: TypeAlias = int


@dataclass(frozen=True)
class StructString:
    """
    a struct String instance.


    ```go
        // String is the runtime representation of a string.
        // It cannot be used safely or portably and its representation may
        // change in a later release.
        //
        // Unlike reflect.StringHeader, its Data field is sufficient to guarantee the
        // data it references will not be garbage collected.
        type String struct {
            Data unsafe.Pointer
            Len  int
        }
    ```

    https://github.com/golang/go/blob/36ea4f9680f8296f1c7d0cf7dbb1b3a9d572754a/src/internal/unsafeheader/unsafeheader.go#L28-L37



    ```rust
        /// # Representation
        ///
        /// A `String` is made up of three components: a pointer to some bytes, a
        /// length, and a capacity. The pointer points to an internal buffer `String`
        /// uses to store its data. The length is the number of bytes currently stored
        /// in the buffer, and the capacity is the size of the buffer in bytes. As such,
        /// the length will always be less than or equal to the capacity.
        ///

    ```
    We only use pointer and length data

    https://github.com/rust-lang/rust/blob/3911a63b7777e19dad4043542f908018e70c0bdd/library/alloc/src/string.rs

    """

    address: VA
    length: int


def get_rdata_section(pe: pefile.PE) -> pefile.SectionStructure:
    for section in pe.sections:
        if section.Name.startswith(b".rdata\x00"):
            return section

    raise ValueError("no .rdata section found")


def get_image_range(pe: pefile.PE) -> Tuple[VA, VA]:
    """return the range of the image in memory."""
    image_base = pe.OPTIONAL_HEADER.ImageBase
    image_size = pe.OPTIONAL_HEADER.SizeOfImage
    return image_base, image_base + image_size


def get_section_name_for_offset(pe: pefile.PE, file_offset: int) -> str:
    """return the name of the section containing the given raw file offset.

    ``StaticString.offset`` is an absolute raw file offset; it must be converted
    to an RVA before querying pefile, rather than being passed directly as an RVA.
    """
    try:
        rva = pe.get_rva_from_offset(file_offset)
    except Exception:
        return ""
    if rva is None:
        return ""

    section = pe.get_section_by_rva(rva)
    if not section:
        return ""
    return section.Name.decode("utf-8").split("\x00")[0]


def _scan_immediates(pattern: "re.Pattern", buf: bytes, fmt: str, base_addr: VA, rip_relative: bool) -> Iterable[VA]:
    """
    scan ``buf`` for instruction pattern ``pattern`` and yield the embedded
    target virtual addresses. ``rip_relative`` instructions add the instruction
    length; all others are absolute, ``base_addr`` is ignored.
    """
    insn_len = 7 if rip_relative else 0
    for match in pattern.finditer(buf):
        value = struct.unpack(fmt, match.group("address"))[0]
        if rip_relative:
            yield base_addr + match.start() + value + insn_len
        else:
            yield value


_AMD64_LEA_XREFS_RE = re.compile(
    # use rb, or else double escape the term "\x0D", or else beware!
    rb"""
    (?:                   # non-capturing group
          \x48 \x8D \x05  # 48 8d 05 aa aa 00 00    lea    rax,[rip+0xaaaa]
        | \x48 \x8D \x0D  # 48 8d 0d aa aa 00 00    lea    rcx,[rip+0xaaaa]
        | \x48 \x8D \x15  # 48 8d 15 aa aa 00 00    lea    rdx,[rip+0xaaaa]
        | \x48 \x8D \x1D  # 48 8d 1d aa aa 00 00    lea    rbx,[rip+0xaaaa]
        | \x48 \x8D \x2D  # 48 8d 2d aa aa 00 00    lea    rbp,[rip+0xaaaa]
        | \x48 \x8D \x35  # 48 8d 35 aa aa 00 00    lea    rsi,[rip+0xaaaa]
        | \x48 \x8D \x3D  # 48 8d 3d aa aa 00 00    lea    rdi,[rip+0xaaaa]
        | \x4C \x8D \x05  # 4c 8d 05 aa aa 00 00    lea     r8,[rip+0xaaaa]
        | \x4C \x8D \x0D  # 4c 8d 0d aa aa 00 00    lea     r9,[rip+0xaaaa]
        | \x4C \x8D \x15  # 4c 8d 15 aa aa 00 00    lea    r10,[rip+0xaaaa]
        | \x4C \x8D \x1D  # 4c 8d 1d aa aa 00 00    lea    r11,[rip+0xaaaa]
        | \x4C \x8D \x25  # 4c 8d 25 aa aa 00 00    lea    r12,[rip+0xaaaa]
        | \x4C \x8D \x2D  # 4c 8d 2d aa aa 00 00    lea    r13,[rip+0xaaaa]
        | \x4C \x8D \x35  # 4c 8d 35 aa aa 00 00    lea    r14,[rip+0xaaaa]
        | \x4C \x8D \x3D  # 4c 8d 3d aa aa 00 00    lea    r15,[rip+0xaaaa]
    )
    (?P<address>....)
    """,
    re.DOTALL | re.VERBOSE,
)

_I386_LEA_XREFS_RE = re.compile(
    rb"""
    (
          \x8D \x05  # 8d 05 aa aa 00 00       lea    eax,ds:0xaaaa
        | \x8D \x1D  # 8d 1d aa aa 00 00       lea    ebx,ds:0xaaaa
        | \x8D \x0D  # 8d 0d aa aa 00 00       lea    ecx,ds:0xaaaa
        | \x8D \x15  # 8d 15 aa aa 00 00       lea    edx,ds:0xaaaa
        | \x8D \x35  # 8d 35 aa aa 00 00       lea    esi,ds:0xaaaa
        | \x8D \x3D  # 8d 3d aa aa 00 00       lea    edi,ds:0xaaaa
    )
    (?P<address>....)
    """,
    re.DOTALL + re.VERBOSE,
)

_PUSH_XREFS_RE = re.compile(
    rb"""
    (
          \x68       # 68 aa aa 00 00       push   0xaaaa
    )
    (?P<address>....)
    """,
    re.DOTALL + re.VERBOSE,
)

_I386_MOV_XREFS_RE = re.compile(
    rb"""
    (
          \xB9       # b9 aa aa 00 00       mov    ecx,0xaaaa
        | \xBB       # bb aa aa 00 00       mov    ebx,0xaaaa
        | \xBA       # ba aa aa 00 00       mov    edx,0xaaaa
        | \xB8       # b8 aa aa 00 00       mov    eax,0xaaaa
        | \xBE       # be aa aa 00 00       mov    esi,0xaaaa
        | \xBF       # bf aa aa 00 00       mov    edi,0xaaaa
    )
    (?P<address>....)
    """,
    re.DOTALL + re.VERBOSE,
)

_AMD64_MOV_XREFS_RE = re.compile(
    rb"""
    (
          \x48 \xC7 \xC0       # 48 c7 c0 aa aa 00 00       mov    rax,0xaaaa
        | \x48 \xC7 \xC1       # 48 c7 c1 aa aa 00 00       mov    rcx,0xaaaa
        | \x48 \xC7 \xC2       # 48 c7 c2 aa aa 00 00       mov    rdx,0xaaaa
        | \x48 \xC7 \xC3       # 48 c7 c3 aa aa 00 00       mov    rbx,0xaaaa
        | \x48 \xC7 \xC5       # 48 c7 c5 aa aa 00 00       mov    rbp,0xaaaa
        | \x48 \xC7 \xC6       # 48 c7 c6 aa aa 00 00       mov    rsi,0xaaaa
        | \x48 \xC7 \xC7       # 48 c7 c7 aa aa 00 00       mov    rdi,0xaaaa
    )
    (?P<address>....)
    """,
    re.DOTALL + re.VERBOSE,
)


def find_amd64_lea_xrefs(buf: bytes, base_addr: VA) -> Iterable[VA]:
    """
    find all the 64-bit RIP-relative LEA instructions,
    extracting the target virtual address.
    """
    yield from _scan_immediates(_AMD64_LEA_XREFS_RE, buf, "<i", base_addr, rip_relative=True)


def find_i386_lea_xrefs(buf: bytes) -> Iterable[VA]:
    """
    find all the 32-bit absolutely addressed LEA instructions,
    extracting the target virtual address.
    """
    yield from _scan_immediates(_I386_LEA_XREFS_RE, buf, "<I", 0, rip_relative=False)


def find_i386_push_xrefs(buf: bytes) -> Iterable[VA]:
    """find all the 32-bit PUSH instructions and their target virtual addresses."""
    yield from _scan_immediates(_PUSH_XREFS_RE, buf, "<I", 0, rip_relative=False)


def find_amd64_push_xrefs(buf: bytes, base_addr: VA) -> Iterable[VA]:
    """find all the 64-bit PUSH instructions and their target virtual addresses."""
    yield from _scan_immediates(_PUSH_XREFS_RE, buf, "<I", base_addr, rip_relative=False)


def find_i386_mov_xrefs(buf: bytes) -> Iterable[VA]:
    """find all the 32-bit MOV instructions and their target virtual addresses."""
    yield from _scan_immediates(_I386_MOV_XREFS_RE, buf, "<I", 0, rip_relative=False)


def find_amd64_mov_xrefs(buf: bytes, base_addr: VA) -> Iterable[VA]:
    """find all the 64-bit MOV instructions and their target virtual addresses."""
    yield from _scan_immediates(_AMD64_MOV_XREFS_RE, buf, "<I", base_addr, rip_relative=False)


def _iter_section_xrefs(pe: pefile.PE, amd64_scan, i386_scan) -> Iterable[VA]:
    """
    scan the executable sections of the given PE file with the arch-specific
    scanner, yielding targets that fall within the image range.
    """
    low, high = get_image_range(pe)

    for section in pe.sections:
        if not section.IMAGE_SCN_MEM_EXECUTE:
            continue

        code = section.get_data()

        if pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
            xrefs = amd64_scan(code, section.VirtualAddress + pe.OPTIONAL_HEADER.ImageBase)
        elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_I386"]:
            xrefs = i386_scan(code)
        else:
            raise ValueError("unhandled architecture")

        for xref in xrefs:
            if low <= xref < high:
                yield xref


def find_lea_xrefs(pe: pefile.PE) -> Iterable[VA]:
    """scan the executable sections for LEA instructions that reference valid addresses."""
    yield from _iter_section_xrefs(pe, find_amd64_lea_xrefs, find_i386_lea_xrefs)


def find_push_xrefs(pe: pefile.PE) -> Iterable[VA]:
    """scan the executable sections for PUSH instructions that reference valid addresses."""
    yield from _iter_section_xrefs(pe, find_amd64_push_xrefs, find_i386_push_xrefs)


def find_mov_xrefs(pe: pefile.PE) -> Iterable[VA]:
    """scan the executable sections for MOV instructions that reference valid addresses."""
    yield from _iter_section_xrefs(pe, find_amd64_mov_xrefs, find_i386_mov_xrefs)


def get_max_section_size(pe: pefile.PE) -> int:
    """get the size of the largest section, as seen on disk."""
    return max(map(lambda s: s.SizeOfRawData, pe.sections))


def get_struct_string_candidates_with_pointer_size(pe: pefile.PE, buf: bytes, psize: int) -> Iterable[StructString]:
    """
    scan through the given bytes looking for pairs of machine words (address, length)
    that might potentially be struct String instances.

    we do some initial validation, like checking that the address is valid
    and the length is reasonable; however, we don't validate the encoded string data.
    """
    if psize == 32:
        format = "I"
    elif psize == 64:
        format = "Q"
    else:
        raise ValueError("unsupported pointer size")

    if not buf:
        return

    limit = get_max_section_size(pe)
    low, high = get_image_range(pe)

    # using array module as a high-performance way to access the data as fixed-sized words.
    words = iter(array.array(format, buf))

    # walk through the words pairwise, (address, length)
    last = next(words)
    for current in words:
        address = last
        length = current
        last = current

        if address == 0x0:
            continue

        if length == 0x0:
            continue

        if length > limit:
            continue

        if not (low <= address < high):
            continue

        yield StructString(address, length)


def get_amd64_struct_string_candidates(pe: pefile.PE, buf: bytes) -> Iterable[StructString]:
    yield from get_struct_string_candidates_with_pointer_size(pe, buf, 64)


def get_i386_struct_string_candidates(pe: pefile.PE, buf: bytes) -> Iterable[StructString]:
    yield from get_struct_string_candidates_with_pointer_size(pe, buf, 32)


def get_struct_string_candidates(pe: pefile.PE) -> Iterable[StructString]:
    """
    find candidate struct String instances in the given PE file.

    we do some initial validation, like checking that the address is valid
    and the length is reasonable; however, we don't validate the encoded string data.
    """
    image_base = pe.OPTIONAL_HEADER.ImageBase
    low, high = get_image_range(pe)

    # cache the section data so that we can avoid pefile overhead
    section_datas: List[Tuple[VA, VA, memoryview]] = []
    for section in pe.sections:
        if not section.IMAGE_SCN_MEM_READ:
            continue

        section_datas.append(
            (
                image_base + section.VirtualAddress,
                image_base + section.VirtualAddress + section.SizeOfRawData,
                # use memoryview here so that we can slice it quickly later
                memoryview(section.get_data()),
            )
        )

    for section in pe.sections:
        if section.IMAGE_SCN_MEM_EXECUTE:
            continue

        if not section.IMAGE_SCN_MEM_READ:
            continue

        # TODO add .text here for Go version 1.12?
        if not (section.Name.startswith(b".rdata\x00") or section.Name.startswith(b".data\x00")):
            # by convention, the struct String instances are stored in the .rdata or .data section.
            continue

        data = section.get_data()

        if pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
            candidates = get_amd64_struct_string_candidates(pe, data)
        elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_I386"]:
            candidates = get_i386_struct_string_candidates(pe, data)
        else:
            raise ValueError("unhandled architecture")

        with floss.utils.timing("find struct string candidates (raw)"):
            candidates = list(candidates)

        for candidate in candidates:
            va = candidate.address
            rva = va - image_base

            if not (low <= va < high):
                continue

            target_section = pe.get_section_by_rva(rva)
            if not target_section:
                # string instance must be in a section
                continue

            # TODO in older Go versions, e.g. 1.12, this may be the case (stored in .text), see 33b5da...
            if target_section.IMAGE_SCN_MEM_EXECUTE:
                # string instances aren't found with the code
                continue

            if not target_section.IMAGE_SCN_MEM_READ:
                # string instances must be readable, naturally
                continue

            try:
                section_start, _, section_data = next(filter(lambda s: s[0] <= candidate.address < s[1], section_datas))
            except StopIteration:
                continue

            instance_offset = candidate.address - section_start
            # remember: section_data is a memoryview, so this is a fast slice.
            # when not using memoryview, this takes a *long* time (dozens of seconds or longer).
            instance_data = section_data[instance_offset : instance_offset + candidate.length]

            if len(instance_data) != candidate.length:
                continue

            yield candidate

            # we would want to be able to validate that structure actually points
            # to valid UTF-8 data;
            # however, even copying the bytes here is very slow,
            # dozens of seconds or more (suspect many minutes).


def get_extract_stats(
    pe: pefile, all_ss_strings: List[StaticString], lang_strings: List[StaticString], min_len: int, min_blob_len=0
) -> float:
    # min_blob_len: this is the minimum length of a string blob in binary file to be considered for extraction
    all_strings = list()
    # these are ascii, extract these utf-8 to get fewer chunks (ascii may split on two-byte characters, for example)
    for ss in all_ss_strings:
        secname = get_section_name_for_offset(pe, ss.offset)
        all_strings.append((secname, ss))

    len_all_ss = 0
    len_lang_str = 0

    # all strings are referenced by absolute raw file offsets; restrict the
    # language strings to .rdata using the same identity as the static strings.
    rdata_lang_strings = [
        lang_str for lang_str in lang_strings if get_section_name_for_offset(pe, lang_str.offset) == ".rdata"
    ]

    # track object identities (StaticString is unhashable and value-equal
    # duplicates must still be treated as distinct entries)
    lang_str_found: Set[int] = set()
    results = list()
    for secname, s in all_strings:
        if secname != ".rdata":
            continue

        if len(s.string) <= min_blob_len:
            continue

        len_all_ss += len(s.string)

        orig_len = len(s.string)
        sha256 = hashlib.sha256()
        sha256.update(s.string.encode("utf-8"))
        s_id = sha256.hexdigest()[:3].upper()
        s_range = get_string_span(s)

        contained = find_contained_lang_strings(s, rdata_lang_strings)
        if not contained:
            null = StaticString(string="", offset=0, encoding=StringEncoding.UTF8)
            results.append((secname, s_id, s_range, False, "", s, 0, null))
            continue

        # account for each language string against its absolute file span
        for lang_str in sorted(contained, key=lambda ls: ls.offset):
            idx = lang_str.offset - s_range[0]
            len_lang_str += len(lang_str.string)

            type_ = "exactsubstr" if idx == 0 and get_string_span(lang_str) == s_range else "substring"

            # remaining character count after this language string is removed
            # from the original span (the span identity itself is byte-based)
            remaining_len = orig_len - len(lang_str.string)
            results.append((secname, s_id, s_range, True, type_, s, remaining_len, lang_str))
            lang_str_found.add(id(lang_str))

        # remove every contained span from the original raw bytes in one pass,
        # keeping each surviving piece at its own absolute file offset.
        pieces, _ = cut_contained_lang_spans(s, contained)
        surviving = "".join(piece.string for piece in pieces)

        if len(surviving) < min_len:
            results.append(
                (
                    secname,
                    s_id,
                    s_range,
                    False,
                    "missing",
                    StaticString(string=surviving, offset=s_range[0], encoding=s.encoding),
                    orig_len - len(surviving),
                    contained[-1],
                )
            )

    rows = list()
    for lang_str in rdata_lang_strings:
        lang_str_sec = get_section_name_for_offset(pe, lang_str.offset)
        if lang_str_sec != ".rdata":
            continue

        if id(lang_str) in lang_str_found:
            continue

        lang_str_data = lang_str.string
        if len(lang_str.string) >= 50:
            lang_str_data = lang_str.string[:36] + "...." + lang_str.string[-10:]
        lang_str_data = sanitize(lang_str_data)

        rows.append(
            (
                f"{lang_str_sec}",
                f"",
                f"",
                f"{lang_str.offset:8x}",
                f"",
                f"unmatched Language string",
                f"",
                f"",
                f"{len(lang_str.string) if lang_str.string else ''}",
                f"{lang_str_data}",
                f"{hex(lang_str.offset) if lang_str.offset else ''}",
            )
        )

    for r in results:
        secname, s_id, s_range, found, msg, s, len_after, lang_str = r

        sdata = s.string
        if len(s.string) >= 50:
            sdata = s.string[:36] + "...." + s.string[-10:]
        sdata = sanitize(sdata)

        lang_str_data = lang_str.string
        if len(lang_str.string) >= 50:
            lang_str_data = lang_str.string[:36] + "...." + lang_str.string[-10:]
        lang_str_data = sanitize(lang_str_data)

        len_info = f"{len(s.string):3d}"
        if found:
            len_info = f"{len(s.string):3d} > {len_after:3d} ({(len(s.string) - len_after) * -1:2d})"

        rows.append(
            (
                f"{secname}",
                f"<{s_id}>",
                f"{s_range[0]:x} - {s_range[1]:x}",
                f"{s.offset:8x}",
                f"{found}",
                f"{msg}",
                len_info,
                f"{sdata}",
                f"{len(lang_str.string) if lang_str.string else ''}",
                f"{lang_str_data}",
                f"{hex(lang_str.offset) if lang_str.offset else ''}",
            )
        )

    rows = sorted(rows, key=lambda t: t[3])

    print(
        tabulate.tabulate(
            rows,
            headers=[
                "section",
                "id",
                "range",
                "offset",
                "found",
                "msg",
                "slen",
                "string",
                "lang_str_len",
                "lang_string",
                "lang_str_off",
            ],
            tablefmt="psql",
        )
    )

    print(".rdata only")
    print("len all string chars:", len_all_ss)
    print("len lang string chars  :", len_lang_str)
    print(f"Percentage of string chars extracted: {round(100 * (len_lang_str / len_all_ss))}%")
    print()

    return 100 * (len_lang_str / len_all_ss)


def strip_found_lang_string(s: StaticString, lang_str: StaticString) -> Tuple[StaticString, int]:
    """remove the found language string from ``s``, returning the trimmed
    ``StaticString`` and its remaining length.

    the language string must reference the *same* bytes as ``s`` in the file,
    i.e. its absolute file-offset span must be contained in the span of ``s``.
    the bytes covered by ``lang_str`` are located via their shared file identity
    (offsets), never via a content search, so that identical content elsewhere in
    the file cannot be stripped or mis-attributed.
    """
    start, end = get_string_span(s)
    lang_start, lang_end = get_string_span(lang_str)

    if not (start <= lang_start and lang_end <= end):
        raise ValueError("language string span is not contained in the static string span")

    s_bytes = s.string.encode("utf-8")
    lang_bytes = lang_str.string.encode("utf-8")

    # relative position of the language bytes inside the static string bytes
    idx = lang_start - start
    length = lang_end - lang_start

    if s_bytes[idx : idx + length] != lang_bytes:
        raise ValueError("language string bytes do not match the static string bytes at the shared offset")

    replaced_bytes = s_bytes[:idx] + s_bytes[idx + length :]
    replaced_s = _decode_piece(replaced_bytes)
    replaced_len = len(replaced_s)

    # keep the absolute file identity of the surviving bytes:
    # - with a surviving prefix, the string still starts at the original offset
    # - otherwise the string now starts immediately after the removed language span
    new_offset = start if idx > 0 else lang_end

    return StaticString(string=replaced_s, offset=new_offset, encoding=s.encoding), replaced_len


def get_string_span(s: StaticString) -> Tuple[int, int]:
    """return the absolute, half-open file-offset span ``[start, end)`` of a string.

    all static/language strings are identified by their raw file offset and the
    UTF-8 byte length of their content, so candidates, searched ranges, extracted
    strings and missed-string consumers share one absolute file identity.
    """
    start = s.offset
    return start, start + len(s.string.encode("utf-8"))


def is_span_contained(inner: StaticString, outer: StaticString) -> bool:
    """return True if the absolute file-offset span of ``inner`` is fully
    contained in the span of ``outer`` (identical spans count as contained)."""
    inner_start, inner_end = get_string_span(inner)
    outer_start, outer_end = get_string_span(outer)
    return outer_start <= inner_start and inner_end <= outer_end


def find_contained_lang_strings(s: StaticString, lang_strings: List[StaticString]) -> List[StaticString]:
    """return all language strings whose absolute file-offset span is contained
    in the span of static string ``s``.

    matching is driven solely by the shared absolute file identity, so a language
    string whose offset merely lands *near* (but not inside) the candidate span
    cannot strip same-named content from a different region of the file.
    """
    return [lang_str for lang_str in lang_strings if lang_str.string and is_span_contained(lang_str, s)]


def cut_contained_lang_spans(s: StaticString, lang_strings: List[StaticString]) -> Tuple[List[StaticString], bool]:
    """remove every language-string span contained in ``s`` from its raw bytes.

    the language strings are matched and removed by their absolute file-offset
    spans against the *original* static string bytes, which keeps the identity
    consistent even when several language strings fall inside one static string
    or when removing an earlier span would shift later offsets.

    returns the surviving ``StaticString`` pieces (each carrying its own absolute
    file offset) and whether at least one language span was removed.
    """
    contained = find_contained_lang_strings(s, lang_strings)
    if not contained:
        return [s], False

    start, end = get_string_span(s)
    raw = s.string.encode("utf-8")

    # validate that every claimed span really holds the language bytes at that
    # absolute location, then merge the half-open spans to cut.
    cut_spans: List[Tuple[int, int]] = []
    for lang_str in contained:
        lang_start, lang_end = get_string_span(lang_str)
        idx = lang_start - start
        length = lang_end - lang_start
        if raw[idx : idx + length] != lang_str.string.encode("utf-8"):
            raise ValueError("language string bytes do not match the static string bytes at the shared offset")
        cut_spans.append((lang_start, lang_end))

    cut_spans.sort()
    merged: List[Tuple[int, int]] = []
    for span_start, span_end in cut_spans:
        if merged and span_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span_end))
        else:
            merged.append((span_start, span_end))

    pieces: List[StaticString] = []
    cursor = start
    for span_start, span_end in merged:
        if span_start > cursor:
            piece_bytes = raw[cursor - start : span_start - start]
            pieces.append(StaticString(string=_decode_piece(piece_bytes), offset=cursor, encoding=s.encoding))
        cursor = span_end
    if cursor < end:
        piece_bytes = raw[cursor - start :]
        pieces.append(StaticString(string=_decode_piece(piece_bytes), offset=cursor, encoding=s.encoding))

    return pieces, True


def _decode_piece(piece_bytes: bytes) -> str:
    """decode a surviving byte piece, tolerating a boundary that splits a
    multi-byte UTF-8 sequence on pathological input instead of aborting analysis."""
    try:
        return piece_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return piece_bytes.decode("utf-8", errors="replace")


def get_missed_strings(
    all_ss_strings: List[StaticString], lang_strings: List[StaticString], min_len: int
) -> List[StaticString]:
    missed_strings = list()

    for s in all_ss_strings:
        pieces, found = cut_contained_lang_spans(s, lang_strings)

        if not found:
            missed_strings.append(s)
            continue

        # the language string(s) only covered part of the static string's file
        # span; keep the legitimate surviving pieces as missed evidence, each with
        # its own absolute file offset.
        for piece in pieces:
            if piece.string and len(piece.string) >= min_len:
                missed_strings.append(piece)

    return missed_strings
