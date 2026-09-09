"""CPU-only regression tests; unittest or pytest can run these without ML packages."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reproduce_historical_benchmark as audit

spec = importlib.util.spec_from_file_location("competitor", ROOT / "reproduction/benchmark/run_competitor.py")
competitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(competitor)


class BenchmarkProvenanceTests(unittest.TestCase):
    def test_archived_metric_recomputes_without_claiming_current_label_error(self):
        report, rows = audit.audit()
        self.assertEqual(len(rows), 135)
        self.assertEqual(report["archive_1_4b_spearman_with_released_labels"], 0.8360456283218484)
        self.assertTrue(report["arithmetic_match"])
        self.assertEqual(report["header_label_conflicts"], 133)
        self.assertFalse(report["header_fitness_multiset_equals_label_multiset"])
        self.assertEqual(report["biological_reproduction_status"], "NOT_ASSESSED_ARCHIVED_ARITHMETIC_ONLY")
        self.assertIn("Header metadata is not assay ground truth", report["status_scope"])

    def test_all_snapshot_hashes(self):
        manifest = audit.verify_bundle(audit.BUNDLE)
        self.assertGreater(len(manifest["files"]), 80)
        supplement = json.loads((audit.BUNDLE / "historical_14b_manifest.json").read_text())
        self.assertEqual(len(supplement["files"]), 42)
        for entry in supplement["files"]:
            path = audit.BUNDLE / "historical_14b" / entry["path"]
            self.assertEqual(path.stat().st_size, entry["bytes"])
            self.assertEqual(audit.sha256(path), entry["sha256"])

    def test_hf_config_and_tokenizer_match_historical(self):
        manifest = audit.verify_bundle(audit.BUNDLE)
        identity = manifest["checkpoint_identity"]["verified_file_sha256"]
        for name in ("config.json", "tokenizer.json"):
            self.assertEqual(audit.sha256(audit.BUNDLE / "checkpoint_metadata/21m_mid_86006" / name), identity[name])
        self.assertEqual(identity["model_weights.pt"], "45d56a7399c4936429da149edf5003bbb5999490a64ec7d793bf5ac98116eb01")

    def test_tied_ranks(self):
        self.assertEqual(audit.average_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
        self.assertEqual(audit.spearman([1, 1, 2], [1, 1, 2]), 1.0)

    def test_nonfinite_empty_mismatched_constant_rejected(self):
        for a, b in [([], []), ([1], [1]), ([1, 2], [1]), ([1, math.nan], [1, 2]),
                     ([1, math.inf], [1, 2]), ([2, 2], [1, 2])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                audit.spearman(a, b)

    def test_archive_join_by_id_not_position(self):
        records = [{"id": "a", "sequence": "AU"}, {"id": "b", "sequence": "CG"}]
        raw = {"scores": [{"header": "b", "sequence": "CG", "log_likelihood": 2},
                          {"header": "a", "sequence": "AT", "log_likelihood": 1}]}
        self.assertEqual(audit.align_archive(records, raw), [1, 2])

    def test_archive_wrong_sequence_rejected(self):
        with self.assertRaises(ValueError):
            audit.align_archive([{"id": "a", "sequence": "AU"}], {"scores": [{"header": "a", "sequence": "AC", "log_likelihood": 1}]})

    def test_archive_duplicate_id_rejected(self):
        row = {"header": "a", "sequence": "AU", "log_likelihood": 1}
        with self.assertRaises(ValueError):
            audit.align_archive([{"id": "a", "sequence": "AU"}, {"id": "b", "sequence": "AU"}], {"scores": [row, row]})

    def test_default_cli_nonzero_despite_arithmetic_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_historical_benchmark.py"), "--output", str(Path(tmp) / "out")], capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr.decode())
            self.assertTrue((Path(tmp) / "out/sequence_label_audit.csv").exists())

    def test_artifact_only_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, str(ROOT / "scripts/reproduce_historical_benchmark.py"), "--output", str(Path(tmp) / "out"), "--artifact-only"]
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_evaluate_requires_same_ids_and_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, l = Path(tmp) / "p.csv", Path(tmp) / "l.csv"
            p.write_text("variant_id,sequence,score\na,AU,1\nb,CG,2\n")
            l.write_text("variant_id,sequence,label\nb,CG,4\na,AT,3\n")
            self.assertEqual(competitor.evaluate(p, l)["spearman"], 1.0)
            l.write_text("variant_id,sequence,label\na,CG,4\nb,AU,3\n")
            with self.assertRaises(ValueError):
                competitor.evaluate(p, l)

    def test_model_manifest_rejects_extra_or_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"; model.mkdir()
            (model / "config.json").write_text("{}")
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"files": competitor.model_files(model)}))
            competitor.verify_model(model, manifest)
            (model / "unexpected.py").write_text("# extra")
            with self.assertRaises(ValueError):
                competitor.verify_model(model, manifest)

    def test_protein_mapping_strict_not_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            dms, ref = Path(tmp) / "assay.csv", Path(tmp) / "reference.csv"
            ref.write_text("DMS_filename,target_seq\nassay.csv,ACD\n")
            dms.write_text("mutant,mutated_sequence,DMS_score\nA1G,GCD,1\nC2A,AAD,2\n")
            rows, wt = competitor.validate_protein(dms, ref)
            self.assertEqual(wt, "ACD"); self.assertEqual(len(rows), 2)
            dms.write_text("mutant,mutated_sequence,DMS_score\nA9G,GCD,1\n")
            with self.assertRaises(ValueError):
                competitor.validate_protein(dms, ref)

    def test_rna_preflight_needs_no_ml_import_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"; model.mkdir()
            (model / "config.json").write_text("{}")
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"files": competitor.model_files(model)}))
            out = Path(tmp) / "out"
            command = [sys.executable, str(ROOT / "reproduction/benchmark/run_competitor.py"), "rna-mlm", "--input", str(audit.BUNDLE / "fixtures/milena/Milena_2021_cata.fasta"), "--model-dir", str(model), "--model-manifest", str(manifest), "--max-tokens", "512", "--output", str(out), "--dry-run"]
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertFalse(out.exists())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Optional CPU forward smoke requires torch/transformers")
    def test_rna_random_tiny_model_forward_smoke_not_paper_reproduction(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "tiny"; model_dir.mkdir()
            vocab = {token: i for i, token in enumerate(["<unk>", "<pad>", "<mask>", "<bos>", "<eos>", "A", "C", "G", "U"])}
            backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = pre_tokenizers.Split("", "isolated")
            backend.post_processor = processors.TemplateProcessing(single="<bos> $A <eos>", special_tokens=[("<bos>", 3), ("<eos>", 4)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", mask_token="<mask>", bos_token="<bos>", eos_token="<eos>")
            tokenizer.save_pretrained(model_dir)
            torch.manual_seed(17)
            model = BertForMaskedLM(BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, max_position_embeddings=32))
            model.save_pretrained(model_dir)
            args = SimpleNamespace(model_dir=model_dir, device="cpu", register_multimolecule=False, trust_local_code=False, sequence_type="rna", max_tokens=6, length_policy="error", mask_batch_size=1)
            records = [{"id": "s1", "sequence": "ACGU"}, {"id": "s2", "sequence": "AAAA"}]
            first = competitor.rna_scores(args, records)
            self.assertEqual([r["scored_tokens"] for r in first], [4, 4])
            args.mask_batch_size = 2
            second = competitor.rna_scores(args, records)
            for a, b in zip(first, second):
                self.assertAlmostEqual(a["score"], b["score"], places=5)
            args.max_tokens = 4
            with self.assertRaises(ValueError):
                competitor.rna_scores(args, records)
            args.length_policy = "truncate"
            self.assertTrue(all(r["truncated"] for r in competitor.rna_scores(args, records)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
