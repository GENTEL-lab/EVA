"""CPU regressions for the archival SAE reproduction entry point."""
import copy
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reproduce_sae", ROOT / "scripts/reproduce_sae.py")
sae = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sae)


def generation_row(condition="no_steer", seed="42", sample="0", context="GCAAGC"):
    active = condition != "no_steer"
    return {
        "case_index": "0", "record_id": "rna_1", "direction": "wt_state",
        "target_state": "wt", "feature_id": "7", "sample_idx": sample,
        "sample_seed": seed, "span_start0": "2", "span_end0": "4",
        "anchor_pos1": "1", "full_sequence": context,
        "structure": "(....)" if active else "......",
        "condition": condition, "scale": "" if not active else "0",
        "target_hits": "1" if active else "0", "opposite_hits": "0",
        "net_target_score": "1" if active else "0",
    }


def likelihood_row(condition, delta):
    return {
        "case_index": 0, "record_id": "rna_1", "direction": "mutant_state",
        "target_state": "mutant", "feature_id": 7,
        "feature_state_margin": .2, "feature_positive_mean": .4,
        "condition": condition, "scale": "" if condition == "no_steer" else 1.,
        "delta_margin": delta, "target_vs_opposite_margin": delta,
        "delta_target_logp": delta, "target_mutation_prob": .5,
        "anchor_one_x": .4, "anchor_pos1": 1,
        "span_start0": 2, "span_end0": 4, "target_span": "AA", "context_span": "UU",
    }


class PairingTests(unittest.TestCase):
    def test_empty_seeds_are_not_paired(self):
        pairs, excluded = sae.pair_generation_rows([generation_row(seed=""), generation_row("0x", seed="")])
        self.assertEqual(pairs, [])
        self.assertEqual(excluded["missing_seed"], 1)

    def test_missing_seeds_are_not_inferred_from_sample_index(self):
        before, after = generation_row(), generation_row("1x")
        del before["sample_seed"]
        del after["sample_seed"]
        pairs, excluded = sae.pair_generation_rows([before, after])
        self.assertFalse(pairs)
        self.assertEqual(excluded["missing_seed"], 1)

    def test_different_seeds_do_not_pair(self):
        pairs, excluded = sae.pair_generation_rows([generation_row(), generation_row("1x", seed="43")])
        self.assertFalse(pairs)
        self.assertEqual(excluded["missing_baseline"], 1)

    def test_same_seed_and_context_pair(self):
        pairs, excluded = sae.pair_generation_rows([generation_row(), generation_row("1x")])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(excluded)

    def test_zero_scale_is_ablation_not_baseline(self):
        pairs, _ = sae.pair_generation_rows([generation_row(), generation_row("0x")])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1]["condition"], "0x")

    def test_same_seed_different_context_does_not_pair(self):
        pairs, excluded = sae.pair_generation_rows([generation_row(), generation_row("1x", context="ACAAGC")])
        self.assertFalse(pairs)
        self.assertEqual(excluded["missing_baseline"], 1)

    def test_same_record_different_constructed_case_does_not_pair(self):
        before, after = generation_row(), generation_row("1x")
        after["case_index"] = "1"
        pairs, excluded = sae.pair_generation_rows([before, after])
        self.assertFalse(pairs)
        self.assertEqual(excluded["missing_baseline"], 1)

    def test_conflicting_baselines_are_not_arbitrarily_chosen(self):
        first, second = generation_row(), generation_row()
        second["structure"] = "(....)"
        second["target_hits"] = second["net_target_score"] = "1"
        pairs, excluded = sae.pair_generation_rows([first, second, generation_row("1x")])
        self.assertFalse(pairs)
        self.assertEqual(excluded["ambiguous_baseline"], 1)

    def test_identical_intervention_is_not_double_counted(self):
        after = generation_row("1x")
        pairs, excluded = sae.pair_generation_rows([generation_row(), after, copy.deepcopy(after)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(excluded["identical_duplicate_intervention"], 1)

    def test_nonfinite_or_missing_scores_fail(self):
        for bad in ("nan", "inf", ""):
            with self.subTest(bad=bad):
                after = generation_row("1x")
                after["net_target_score"] = bad
                with self.assertRaises(ValueError):
                    sae.pair_generation_rows([generation_row(), after])

    def test_bad_score_arithmetic_fails(self):
        after = generation_row("1x")
        after["net_target_score"] = "2"
        with self.assertRaises(ValueError):
            sae.pair_generation_rows([generation_row(), after])


class LikelihoodTests(unittest.TestCase):
    def test_original_first_maximum_tie_rule(self):
        rows = [likelihood_row("no_steer", 0), likelihood_row("0x", .1), likelihood_row("1x", .1)]
        summary = sae.summarize_likelihood(rows)
        self.assertEqual(summary[0]["best_condition"], "0x")
        self.assertEqual(sae.select_original_likelihood(summary), [])

    def test_original_selected_positive_condition(self):
        rows = [likelihood_row("no_steer", 0), likelihood_row("0x", .1), likelihood_row("2x", .2)]
        chosen = sae.select_original_likelihood(sae.summarize_likelihood(rows))
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["best_condition"], "2x")

    def test_original_baseline_best_is_removed(self):
        rows = [likelihood_row("no_steer", 0), likelihood_row("1x", -.1)]
        self.assertEqual(sae.select_original_likelihood(sae.summarize_likelihood(rows)), [])

    def test_missing_delta_is_not_silently_zero(self):
        with self.assertRaises(ValueError):
            sae.summarize_likelihood([likelihood_row("1x", "")])

    def test_mismatched_archived_summary_fails(self):
        summary = sae.summarize_likelihood([likelihood_row("2x", .1)])
        archived = [{k: str(v) for k, v in summary[0].items()}]
        archived[0]["best_condition"] = "5x"
        with self.assertRaises(ValueError):
            sae.compare_summary(summary, archived)


class ArtifactTests(unittest.TestCase):
    def test_dot_bracket_validation(self):
        self.assertEqual(sae.structure_pairs("((..))"), {(1, 6), (2, 5)})
        for invalid in ("(()", "())", "[..]"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    sae.structure_pairs(invalid)

    def test_bundled_inputs_and_eight_author_choices(self):
        manifest = sae.verify_bundle(sae.DEFAULT_BUNDLE)
        protocol = sae.read_json(sae.DEFAULT_BUNDLE / "protocol.json")
        with tempfile.TemporaryDirectory() as directory:
            report = sae.examples(sae.DEFAULT_BUNDLE, Path(directory), protocol, manifest)
        self.assertEqual(report["n"], 8)
        self.assertTrue(all(r["wt_hit_delta"] > 0 for r in report["cases"]))
        self.assertEqual(sum(r["net_target_delta"] < 0 for r in report["cases"]), 6)

    def test_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sae.main(["--output", directory, "--mode", "table"])

    def test_archived_table_rows_are_reconstructed_not_raw_recomputed(self):
        protocol = sae.read_json(sae.DEFAULT_BUNDLE / "protocol.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = sae.archived_table(sae.DEFAULT_BUNDLE, output, protocol)
            self.assertEqual((output / "table_s30.reconstructed.tex").read_bytes(),
                             (output / "table_s30.archived.tex").read_bytes())
        self.assertEqual((report["archived_success_n"], report["n"]), (213, 225))
        self.assertIn("not_raw_success_recomputed", report["status"])


if __name__ == "__main__":
    unittest.main()
