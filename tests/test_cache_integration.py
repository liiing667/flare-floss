import hashlib
import json
import logging
from pathlib import Path

from fixtures import scfile, exefile

import floss.main
import floss.cache
import floss.render.json
from floss.results import Strings, Analysis, Metadata, ResultDocument
from floss.version import __version__


def cache_entries(cache_dir):
    return list(cache_dir.rglob("*.json"))


def test_cache_hit_skips_analysis_and_matches_output(capsys, caplog, tmp_path, monkeypatch, exefile):
    caplog.set_level(logging.DEBUG)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))

    assert floss.main.main([exefile, "--summary"]) == 0
    out1 = capsys.readouterr().out

    sha256 = hashlib.sha256(Path(exefile).read_bytes()).hexdigest()
    key = floss.cache.compute_key(sha256, __version__)
    assert (cache_dir / f"{key}.json").is_file()
    assert len(cache_entries(cache_dir)) == 1

    caplog.clear()
    assert floss.main.main([exefile, "--summary", "-d"]) == 0
    assert any("using cached results" in r.getMessage() for r in caplog.records)
    assert capsys.readouterr().out == out1


def test_cache_disabled_writes_nothing(capsys, tmp_path, monkeypatch, exefile):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))
    monkeypatch.setenv(floss.cache.ENV_CACHE_ENABLE, "0")

    assert floss.main.main([exefile, "--summary"]) == 0
    assert cache_entries(cache_dir) == []


def test_cache_not_written_for_results_document(capsys, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))

    doc = ResultDocument(
        metadata=Metadata(file_path="sample.exe", min_length=4),
        analysis=Analysis(enable_static_strings=False),
        strings=Strings(),
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(floss.render.json.render(doc))

    assert floss.main.main([str(results_path)]) == 0
    assert cache_entries(cache_dir) == []


def test_cache_refresh_reanalyzes_and_overwrites(capsys, caplog, tmp_path, monkeypatch, exefile):
    caplog.set_level(logging.DEBUG)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))

    assert floss.main.main([exefile, "--summary"]) == 0
    sha256 = hashlib.sha256(Path(exefile).read_bytes()).hexdigest()
    key = floss.cache.compute_key(sha256, __version__)
    assert (cache_dir / f"{key}.json").is_file()

    # FLOSS_CACHE_REFRESH=1 forces a miss, re-analyzes, and overwrites the entry
    caplog.clear()
    monkeypatch.setenv(floss.cache.ENV_CACHE_REFRESH, "1")
    assert floss.main.main([exefile, "--summary", "-d"]) == 0
    assert not any("using cached results" in r.getMessage() for r in caplog.records)
    assert any("FLOSS_CACHE_REFRESH" in r.getMessage() for r in caplog.records)
    assert (cache_dir / f"{key}.json").is_file()


def test_analysis_variants_get_separate_cache_entries(capsys, caplog, tmp_path, monkeypatch, scfile):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))

    # an explicit --format is a non-default analysis variant: it is still
    # cached, but in a variant-specific entry that never collides with others
    caplog.set_level(logging.DEBUG)
    assert floss.main.main([scfile, "-f", "sc32", "--summary"]) == 0
    entries = cache_entries(cache_dir)
    assert len(entries) == 1
    sc32_entry = entries[0]

    # the same bytes under sc64 get their own entry instead of reusing sc32's
    caplog.clear()
    assert floss.main.main([scfile, "-f", "sc64", "--summary"]) == 0
    assert not any("using cached results" in r.getMessage() for r in caplog.records)
    entries = cache_entries(cache_dir)
    assert len(entries) == 2
    assert sc32_entry in entries

    # re-running sc32 hits its own entry, not the sc64 document
    caplog.clear()
    assert floss.main.main([scfile, "-f", "sc32", "--summary"]) == 0
    assert any("using cached results" in r.getMessage() for r in caplog.records)
    assert len(cache_entries(cache_dir)) == 2


def test_language_variants_do_not_share_a_cache_entry(capsys, caplog, tmp_path, monkeypatch, exefile):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))
    caplog.set_level(logging.DEBUG)

    # force the Go entry point first, then the Rust entry point on the same
    # sample: the second run must analyze and store its own document
    assert floss.main.main([exefile, "--language", "go", "--summary"]) == 0
    go_entries = cache_entries(cache_dir)
    assert len(go_entries) == 1

    caplog.clear()
    assert floss.main.main([exefile, "--language", "rust", "--summary"]) == 0
    assert not any("using cached results" in r.getMessage() for r in caplog.records)
    all_entries = cache_entries(cache_dir)
    assert len(all_entries) == 2

    # alternating back to Go must hit the Go entry and report Go evidence,
    # never the Rust variant's language metadata or strings
    caplog.clear()
    assert floss.main.main([exefile, "--language", "go", "--summary", "-j"]) == 0
    assert any("using cached results" in r.getMessage() for r in caplog.records)

    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["cache_language"] == "go"
    assert len(cache_entries(cache_dir)) == 2


def test_analyze_functions_variant_is_cached_separately(capsys, caplog, tmp_path, monkeypatch, exefile):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))
    caplog.set_level(logging.DEBUG)

    assert floss.main.main([exefile, "--analyze-functions", "0x401560"]) == 0
    entries = cache_entries(cache_dir)
    assert len(entries) == 1

    # a different function selection is a different variant: no cross-hit
    caplog.clear()
    assert floss.main.main([exefile, "--analyze-functions", "0x401560", "0x401000"]) == 0
    assert not any("using cached results" in r.getMessage() for r in caplog.records)
    assert len(cache_entries(cache_dir)) == 2

    # repeating the exact selection (in any order) hits the same entry
    caplog.clear()
    assert floss.main.main([exefile, "--analyze-functions", "0x401000", "0x401560"]) == 0
    assert any("using cached results" in r.getMessage() for r in caplog.records)
    assert len(cache_entries(cache_dir)) == 2


def test_default_variant_keeps_legacy_cache_key(capsys, tmp_path, monkeypatch, exefile):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(floss.cache.ENV_CACHE_DIR, str(cache_dir))

    assert floss.main.main([exefile, "--summary"]) == 0
    sha256 = hashlib.sha256(Path(exefile).read_bytes()).hexdigest()
    # the default variant keeps the exact legacy key shape
    legacy_key = floss.cache.compute_key(sha256, __version__)
    assert (cache_dir / f"{legacy_key}.json").is_file()
