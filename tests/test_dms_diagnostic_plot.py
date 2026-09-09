"""CPU-only tests for plotting already-exported predictions, never model inference."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('dms_plot', Path(__file__).resolve().parents[1] / 'scripts/plot_dms_predictions.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class DiagnosticPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'predictions.csv'
        self.rows = [[9, 'second', 'ACG', 2, -1], [3, 'first', 'UGC', 1, -2]]

    def write_csv(self, rows=None, header=MOD.REQUIRED):
        with self.source.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.rows if rows is None else rows)

    def test_valid_input_generates_all_outputs_without_changing_source_or_order(self):
        self.write_csv()
        before = self.source.read_bytes()
        out = self.root / 'plot'
        report = MOD.plot_predictions(self.source, out)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(report['input_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual([r['source_row'] for r in report['rows']], [9, 3])
        self.assertEqual(report['n'], 2)
        self.assertAlmostEqual(report['spearman'], 1.0)
        self.assertFalse(report['model_inference_performed'])
        self.assertEqual(report['rows_filtered'], 0)
        self.assertEqual(set(p.name for p in out.iterdir()), {'diagnostic_predictions.png', 'diagnostic_predictions.svg', 'report.json'})
        self.assertEqual(json.loads((out / 'report.json').read_text())['status'], report['status'])
        self.assertIn('Diagnostic plot of supplied predictions', (out / 'diagnostic_predictions.svg').read_text())

    def test_existing_output_is_never_overwritten(self):
        self.write_csv()
        out = self.root / 'existing'
        out.mkdir()
        marker = out / 'keep.txt'
        marker.write_text('unchanged')
        with self.assertRaises(FileExistsError):
            MOD.plot_predictions(self.source, out)
        self.assertEqual(marker.read_text(), 'unchanged')

    def test_invalid_headers_are_rejected(self):
        for header in (('sequence_id',), MOD.REQUIRED + ('sequence_id',)):
            with self.subTest(header=header):
                self.write_csv(header=header)
                with self.assertRaises(ValueError):
                    MOD.read_predictions(self.source)

    def test_duplicate_ids_are_rejected(self):
        self.rows[1][1] = self.rows[0][1]
        self.write_csv()
        with self.assertRaises(ValueError):
            MOD.read_predictions(self.source)

    def test_nonfinite_values_are_rejected_without_output(self):
        for column in (3, 4):
            for value in ('NaN', 'Inf', '-Inf'):
                with self.subTest(column=column, value=value):
                    rows = [r[:] for r in self.rows]
                    rows[0][column] = value
                    self.write_csv(rows)
                    with self.assertRaises(ValueError):
                        MOD.plot_predictions(self.source, self.root / 'invalid')
                    self.assertFalse((self.root / 'invalid').exists())

    def test_empty_single_or_constant_data_are_rejected(self):
        cases = [[], self.rows[:1]]
        for column in (3, 4):
            rows = [r[:] for r in self.rows]
            rows[1][column] = rows[0][column]
            cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows):
                self.write_csv(rows)
                with self.assertRaises(ValueError):
                    MOD.read_predictions(self.source)

    def test_malformed_rows_and_empty_fields_are_rejected(self):
        for rows in ([self.rows[0], self.rows[1][:-1]], [self.rows[0], self.rows[1] + ['extra']], [self.rows[0], [3, '', 'UGC', 1, -2]], [self.rows[0], [9, 'first', 'UGC', 1, -2]]):
            with self.subTest(rows=rows):
                self.write_csv(rows)
                with self.assertRaises(ValueError):
                    MOD.read_predictions(self.source)


if __name__ == '__main__':
    unittest.main()
