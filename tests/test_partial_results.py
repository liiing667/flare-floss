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

import json
from typing import Optional
from pathlib import Path

import pytest
from fixtures import exefile  # noqa: F401

import floss.main
import floss.results
import floss.pipeline
from floss.cli import WorkspaceLoadError
from floss.results import Analysis
from floss.pipeline import Options, analyze, PipelineError

SIGS = Path(floss.main.get_default_root()) / "sigs"


def _options(sample: str, analysis: Optional[Analysis] = None, cache_dir: Optional[Path] = None) -> Options:
    return Options(
        sample=Path(sample),
        min_length=4,
        analysis=analysis or Analysis(),
        signatures=SIGS,
        quiet=True,
        cache_dir=cache_dir,
    )


def _fail_load_vw(*args, **kwargs):
    raise WorkspaceLoadError("injected workspace load failure")


def test_workspace_load_failure_recovers_partial_results(monkeypatch, tmp_path, exefile):
    # the workspace load fails after the static strings were extracted: the
    # shared transaction carries the partial results, the failed stage, and
    # the cache publication state instead of losing them
    monkeypatch.setattr(floss.pipeline, "load_vw", _fail_load_vw)

    with pytest.raises(PipelineError) as exc_info:
        analyze(_options(exefile, cache_dir=tmp_path))

    error = exc_info.value
    assert error.stage == floss.pipeline.STAGE_VIVISECT
    assert "injected workspace load failure" in str(error)

    txn = error.transaction
    assert txn is not None
    assert txn.partial
    assert txn.failed_stage == floss.pipeline.STAGE_VIVISECT
    assert floss.pipeline.STAGE_STATIC_STRINGS in txn.accepted_stages
    # partial results are never published to the cache
    assert not txn.cache_published

    results = error.results
    assert results is not None
    # the static evidence gathered before the failure is preserved
    assert results.strings.static_strings
    assert results.layout is not None
    # the stages that never ran are recorded as disabled
    assert results.analysis.enable_static_strings
    assert not results.analysis.enable_stack_strings
    assert not results.analysis.enable_tight_strings
    assert not results.analysis.enable_decoded_strings
    assert results.metadata.runtime.total > 0

    # nothing was published to the cache
    assert list(tmp_path.rglob("*.json")) == []


def test_later_stage_failure_preserves_accepted_stages(monkeypatch, tmp_path, exefile):
    # a real failure in a later stage keeps the stages accepted so far
    def fail_stackstrings(*args, **kwargs):
        raise RuntimeError("injected stackstrings failure")

    monkeypatch.setattr(floss.pipeline, "extract_stackstrings", fail_stackstrings)

    with pytest.raises(PipelineError) as exc_info:
        analyze(_options(exefile, cache_dir=tmp_path))

    txn = exc_info.value.transaction
    assert txn is not None
    assert txn.failed_stage == floss.pipeline.STAGE_STACK_STRINGS
    assert floss.pipeline.STAGE_VIVISECT in txn.accepted_stages
    assert floss.pipeline.STAGE_FIND_FEATURES in txn.accepted_stages

    results = exc_info.value.results
    assert results.strings.static_strings
    assert not results.analysis.enable_stack_strings
    assert list(tmp_path.rglob("*.json")) == []


def test_pure_static_analysis_ignores_workspace_failure(monkeypatch, exefile):
    # pure static analysis never touches the workspace: it is unaffected
    monkeypatch.setattr(floss.pipeline, "load_vw", _fail_load_vw)
    static_only = Analysis(
        enable_stack_strings=False,
        enable_tight_strings=False,
        enable_decoded_strings=False,
    )
    results = analyze(_options(exefile, analysis=static_only))
    assert results.strings.static_strings


def test_full_success_publishes_cache_via_transaction(tmp_path, exefile):
    # the shared transaction records full stage acceptance and cache publication
    txn = floss.pipeline.analyze_transaction(_options(exefile, cache_dir=tmp_path))
    assert not txn.partial
    assert txn.failed_stage is None
    assert txn.cache_published
    assert list(tmp_path.rglob("*.json"))
    for stage in (
        floss.pipeline.STAGE_STATIC_STRINGS,
        floss.pipeline.STAGE_VIVISECT,
        floss.pipeline.STAGE_FIND_FEATURES,
        floss.pipeline.STAGE_STACK_STRINGS,
        floss.pipeline.STAGE_TIGHT_STRINGS,
        floss.pipeline.STAGE_DECODED_STRINGS,
    ):
        assert stage in txn.accepted_stages


def test_cli_json_emits_partial_document_and_stage_error(monkeypatch, capsys, exefile):
    # JSON consumers get the partial document on STDOUT and the failure stage
    # in the structured error on STDERR
    monkeypatch.setattr(floss.pipeline, "load_vw", _fail_load_vw)

    assert floss.main.main([exefile, "-j", "-q"]) == -1

    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["strings"]["static_strings"]
    assert doc["analysis"]["enable_static_strings"]
    assert not doc["analysis"]["enable_stack_strings"]
    assert not doc["analysis"]["enable_tight_strings"]
    assert not doc["analysis"]["enable_decoded_strings"]

    errors = [json.loads(line) for line in captured.err.splitlines() if line.startswith("{")]
    assert any(e.get("stage") == "vivisect" and "failed to analyze sample" in e.get("error", "") for e in errors)


def test_cli_text_renders_partial_results(monkeypatch, capsys, exefile):
    monkeypatch.setattr(floss.pipeline, "load_vw", _fail_load_vw)

    assert floss.main.main([exefile, "-q"]) == -1

    captured = capsys.readouterr()
    # the partial static evidence is rendered, not lost
    assert captured.out


def test_partial_document_round_trips(monkeypatch, tmp_path, capsys, exefile):
    monkeypatch.setattr(floss.pipeline, "load_vw", _fail_load_vw)

    assert floss.main.main([exefile, "-j", "-q"]) == -1
    out = capsys.readouterr().out

    path = tmp_path / "partial.json"
    path.write_text(out)
    doc = floss.results.read(path)
    assert doc.strings.static_strings
    assert not doc.analysis.enable_stack_strings
