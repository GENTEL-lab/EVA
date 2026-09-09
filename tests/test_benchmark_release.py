import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('benchmark_release', ROOT / 'scripts/benchmark_release.py')
BR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BR)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = [['a', 'AAA', 1], ['b', 'AAC', 2], ['c', 'AAG', 3], ['d', 'AAU', 4]]
        self.write('labels.csv', ['variant_id', 'sequence', 'label'], self.labels)
        self.write('scores.csv', ['variant_id', 'sequence', 'score'], list(reversed(self.labels)))
        self.manifest = {'schema_version': 1, 'models': ['test-model'],
                         'datasets': [{'id': 'test-data', 'sequence_policy': 'exact', 'source_status': 'verified',
                                       'labels': self.ref('labels.csv')}],
                         'predictions': [{'dataset': 'test-data', 'model': 'test-model',
                                          'file': self.ref('scores.csv'), 'published_spearman': '1.0000'}]}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, headers, rows):
        with (self.root / name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)

    def ref(self, name):
        return {'path': name, 'sha256': BR.sha256(self.root / name)}

    def evaluate(self):
        return BR.evaluate_manifest(self.manifest, self.root)

    def replace_scores(self, rows):
        self.write('scores.csv', ['variant_id', 'sequence', 'score'], rows)
        self.manifest['predictions'][0]['file'] = self.ref('scores.csv')

    def test_row_order_independence(self):
        report = self.evaluate()
        self.assertTrue(report['complete'])
        self.assertEqual(report['results'][0]['spearman'], 1)
        self.assertFalse(report['paper_reproduction_claim'])

    def test_duplicate_variant_rejected(self):
        self.replace_scores(self.labels + [self.labels[0]])
        self.assertIn('duplicate', self.evaluate()['failures'][0]['error'])

    def test_missing_variant_rejected(self):
        self.replace_scores(self.labels[:-1])
        self.assertIn('ID sets differ', self.evaluate()['failures'][0]['error'])

    def test_extra_variant_rejected(self):
        self.replace_scores(self.labels + [['e', 'ACA', 5]])
        self.assertFalse(self.evaluate()['complete'])

    def test_sequence_mismatch_rejected(self):
        self.replace_scores([['a', 'CCC', 1], *self.labels[1:]])
        self.assertIn('sequence mismatch', self.evaluate()['failures'][0]['error'])

    def test_nan_rejected(self):
        self.replace_scores([['a', 'AAA', 'nan'], *self.labels[1:]])
        self.assertIn('nonfinite', self.evaluate()['failures'][0]['error'])

    def test_inf_rejected(self):
        self.replace_scores([['a', 'AAA', 'inf'], *self.labels[1:]])
        self.assertFalse(self.evaluate()['complete'])

    def test_constant_scores_rejected(self):
        self.replace_scores([[key, seq, 1] for key, seq, _ in self.labels])
        self.assertIn('constant', self.evaluate()['failures'][0]['error'])

    def test_checksum_change_rejected(self):
        (self.root / 'scores.csv').write_text('modified')
        self.assertIn('SHA256 mismatch', self.evaluate()['failures'][0]['error'])

    def test_missing_hash_rejected(self):
        for missing in ('', None):
            with self.subTest(hash=missing):
                self.manifest['predictions'][0]['file']['sha256'] = missing
                self.assertFalse(self.evaluate()['complete'])

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            BR.verified_file(self.root, {'path': '../x', 'sha256': '0'*64})

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            BR.verified_file(self.root, {'path': str(self.root / 'labels.csv'), 'sha256': '0'*64})

    def test_symlink_escape_rejected(self):
        (self.root / 'escape').symlink_to(self.root.parent)
        with self.assertRaises(ValueError):
            BR.verified_file(self.root, {'path': 'escape/nonexistent', 'sha256': '0'*64})

    def test_missing_model_does_not_create_subset_mean(self):
        self.manifest['models'].append('missing-model')
        report = self.evaluate()
        self.assertFalse(report['complete'])
        self.assertIsNone(report['model_summary'][1]['mean_abs_spearman'])

    def test_missing_dataset_disables_partial_mean(self):
        item = copy.deepcopy(self.manifest['datasets'][0])
        item['id'] = 'second-data'
        self.manifest['datasets'].append(item)
        self.assertIsNone(self.evaluate()['model_summary'][0]['mean_abs_spearman'])

    def test_duplicate_job_rejected(self):
        self.manifest['predictions'].append(self.manifest['predictions'][0])
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_unknown_job_rejected(self):
        self.manifest['predictions'][0]['model'] = 'unknown'
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_signed_rho_not_flipped(self):
        self.replace_scores([[key, seq, -value] for key, seq, value in self.labels])
        report = self.evaluate()
        self.assertEqual(report['results'][0]['spearman'], -1)
        self.assertEqual(report['model_summary'][0]['mean_abs_spearman'], 1)
        self.assertFalse(report['all_match_published_precision'])

    def test_ties_average_rank(self):
        self.assertEqual(BR.ranks([1, 2, 2, 4]), [1, 2.5, 2.5, 4])
        self.assertAlmostEqual(BR.spearman([1, 2, 2, 4], [1, 2, 3, 4]), 3 / math.sqrt(10))

    def test_t_to_u_requires_explicit_rna_policy(self):
        self.replace_scores([*self.labels[:3], ['d', 'AAT', 4]])
        self.assertFalse(self.evaluate()['complete'])
        self.manifest['datasets'][0]['sequence_policy'] = 'rna_t_to_u'
        self.assertTrue(self.evaluate()['complete'])

    def test_unknown_sequence_policy_rejected(self):
        self.manifest['datasets'][0]['sequence_policy'] = 'guess'
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_unresolved_source_retained(self):
        self.manifest['datasets'][0]['source_status'] = 'unresolved'
        report = self.evaluate()
        self.assertEqual(report['unresolved_label_sources'], ['test-data'])
        self.assertFalse(report['paper_reproduction_claim'])

    def test_decimal_string_required(self):
        self.manifest['predictions'][0]['published_spearman'] = 1.0
        self.assertIn('decimal string', self.evaluate()['failures'][0]['error'])

    def test_missing_reference_not_marked_match(self):
        self.manifest['predictions'][0]['published_spearman'] = None
        self.assertFalse(self.evaluate()['all_match_published_precision'])

    def test_invalid_csv_header_rejected(self):
        self.write('bad.csv', ['variant_id', 'sequence', 'score', 'score'], [['x', 'AAA', 1, 2]])
        with self.assertRaises(ValueError):
            BR.read_csv(self.root / 'bad.csv', ['score'])

    def test_reference_coverage(self):
        config = json.loads((ROOT / 'examples/reproduction/benchmark_release/reference_manifest.json').read_text())
        report = BR.audit_reference(config, ROOT / 'examples/reproduction/benchmark_release/reference')
        self.assertEqual([(g['observed_pairs'], g['expected_pairs']) for g in report['groups']], [(195, 195), (75, 75), (220, 240)])
        self.assertEqual({r['model'] for r in report['groups'][2]['missing_pairs']}, {'Evo2 7B'})
        self.assertFalse(report['complete'])

    def test_sample_count_discrepancies_exposed(self):
        config = json.loads((ROOT / 'examples/reproduction/benchmark_release/reference_manifest.json').read_text())
        report = BR.audit_reference(config, ROOT / 'examples/reproduction/benchmark_release/reference')
        self.assertEqual(len(report['groups'][0]['inconsistent_sample_counts']), 4)
        self.assertTrue(report['groups'][0]['coverage_complete'])
        self.assertFalse(report['groups'][0]['sample_counts_consistent'])

    def test_long_and_wide_metric_exports(self):
        BR.write_metric_tables(self.evaluate(), self.manifest, self.root)
        _, rows = BR.read_csv(self.root / 'metrics_long.csv', ['Dataset', 'Model', 'Spearman', 'N_samples'])
        self.assertEqual(float(rows[0]['Spearman']), 1)
        columns, rows = BR.read_csv(self.root / 'metrics_wide.csv', ['Model', 'test-data'])
        self.assertEqual(float(rows[0]['test-data']), 1)

    def test_incomplete_metrics_not_exported(self):
        self.manifest['models'].append('missing')
        BR.write_metric_tables(self.evaluate(), self.manifest, self.root)
        self.assertFalse((self.root / 'metrics_long.csv').exists())

    def test_svg_is_parseable(self):
        BR.plot_report(self.evaluate(), self.root)
        root = ET.parse(self.root / 'summary_0.svg').getroot()
        self.assertTrue(root.tag.endswith('svg'))

    def test_cli_and_existing_output(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest))
        command = [sys.executable, str(ROOT / 'scripts/benchmark_release.py'), 'evaluate', '--manifest', str(path), '--data-root', str(self.root), '--output', str(self.root / 'result'), '--plot']
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        run2 = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run2.returncode, 2)
        self.assertIn('already exists', run2.stderr)

    def test_unresolved_cli_exit_two(self):
        self.manifest['datasets'][0]['source_status'] = 'unresolved'
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest))
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/benchmark_release.py'), 'evaluate', '--manifest', str(path), '--data-root', str(self.root), '--output', str(self.root / 'result')], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertTrue((self.root / 'result/report.json').exists())


if __name__ == '__main__':
    unittest.main()
