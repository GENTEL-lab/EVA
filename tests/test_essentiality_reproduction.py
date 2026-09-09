"""CPU-only tests for archived essentiality recomputation (no torch required)."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "essentiality_repro", Path(__file__).resolve().parents[1] / "scripts/reproduce_essentiality.py")
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


def source(seq="ACGU", gene="same"):
    return dict(organism="species", gene=gene, locus_tag="locus", nc="NC_1",
                lineage="d__eukaryota", essential=False, sequence=seq)


def position_row():
    row = source()
    row.pop("sequence")
    return dict(row, record_index=0, label=0, position_id="p05", position="5%",
                position_ratio=.05, length=4, insert_pos=1, mutation=repro.MUTATION,
                scoring_condition="lineage_prefixed", wt_ll=-4., mut_ll=-7., delta_ll=3.)


def test_rank_auc_ties_and_order():
    assert repro.auc([0, 1, 0, 1], [0, 1, 1, 2]) == pytest.approx(.875)
    assert repro.auc([0, 1], [1, 1]) == .5
    assert repro.auc([0, 1], [1, 0]) == 0


@pytest.mark.parametrize("labels,scores", [([], []), ([0], [0]), ([0, 1], [1]),
                                           ([0, 1], [0, float("nan")]), ([0, 2], [0, 1])])
def test_invalid_auc_inputs_fail(labels, scores):
    with pytest.raises(ValueError):
        repro.auc(labels, scores)


def test_same_gene_transcripts_do_not_collapse():
    original = [source("AAAA"), source("GGGG")]
    scored = [{k: v for k, v in r.items() if k != "sequence"} for r in original]
    rows = list(repro.checked_join(original, scored))
    assert rows[0][1] != rows[1][1]
    assert [repro.gc(row[2]["sequence"]) for row in rows] == [0., 1.]


@pytest.mark.parametrize("field,value", [("nc", "NC_2"), ("lineage", "wrong"),
                                         ("essential", True), ("sequence", "GGGG")])
def test_mismatching_join_fails(field, value):
    original = source()
    scored = dict(original, **{field: value})
    with pytest.raises(ValueError, match="mismatch"):
        list(repro.checked_join([original], [scored]))


def test_reordered_records_fail():
    data = [source(gene="one"), source(gene="two")]
    with pytest.raises(ValueError, match="metadata mismatch"):
        list(repro.checked_join(data, list(reversed(data))))


def test_duplicate_position_row_fails():
    seen = set()
    repro.validate_position_row(position_row(), [source()], repro.POSITIONS[0], seen)
    with pytest.raises(ValueError, match="Duplicate"):
        repro.validate_position_row(position_row(), [source()], repro.POSITIONS[0], seen)


@pytest.mark.parametrize("field,value", [("position_id", "p50"), ("position", "50%"),
                                         ("position_ratio", .5), ("delta_ll", 0),
                                         ("wt_ll", float("inf")), ("length", 99),
                                         ("mutation", "UAA"), ("label", 1)])
def test_position_protocol_errors_fail(field, value):
    row = dict(position_row(), **{field: value})
    with pytest.raises(ValueError):
        repro.validate_position_row(row, [source()], repro.POSITIONS[0], set())


def test_gc_ambiguous_bases_keep_historical_denominator():
    assert repro.gc("GCNY") == .5


def test_json_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"p05": 1, "p05": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        repro.load_json(path)


def output_report():
    summary = repro.summarize({"species": [(0, 0), (1, 1)]})
    return dict(position_ablation={model: {p[0]: copy.deepcopy(summary) for p in repro.POSITIONS}
                                    for model in repro.MODELS}, gc_baseline={"corrected": summary})


def test_all_formats_keep_distinct_five_and_fifty_percent(tmp_path):
    report = output_report()
    report["position_ablation"]["eva1400M"]["p05"]["mean_species_auroc"] = .5514
    report["position_ablation"]["eva1400M"]["p50"]["mean_species_auroc"] = .5904
    output = tmp_path / "results"
    repro.write_outputs(output, report)
    json_report = json.loads((output / "report.json").read_text())
    assert len(json_report["position_ablation"]["eva1400M"]) == 5
    rows = list(csv.DictReader((output / "position_ablation.csv").open()))
    assert len(rows) == 10
    assert rows[0]["position_id"] == "p05" and float(rows[0]["mean_species_auroc"]) == .5514
    assert rows[2]["position_id"] == "p50" and float(rows[2]["mean_species_auroc"]) == .5904
    md = (output / "summary.md").read_text()
    assert "5% | 0.5514000000" in md and "50% | 0.5904000000" in md


def test_missing_position_rejected_before_writing(tmp_path):
    report = output_report()
    del report["position_ablation"]["eva1400M"]["p50"]
    with pytest.raises(ValueError, match="Missing or extra"):
        repro.write_outputs(tmp_path / "results", report)
    assert not (tmp_path / "results").exists()


def test_existing_output_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        repro.write_outputs(tmp_path, output_report())
