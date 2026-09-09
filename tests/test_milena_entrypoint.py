import csv
import json
from pathlib import Path
import tempfile
import unittest
from scripts.reproduce_milena import validate_inputs, align_fresh, MANIFEST, ROOT

class MilenaEntrypointTests(unittest.TestCase):
    def test_frozen_current_complete_assay(self):
        manifest, records, labels, archive, reference = validate_inputs()
        self.assertEqual(len(records), 135)
        self.assertEqual(len(labels), len(archive))
        self.assertEqual(manifest['protocol']['reduce'], 'sum')
        self.assertEqual(manifest['checkpoint']['subdirectory'], 'EVA_1.4B_CLM')

    def test_changed_manifest_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(MANIFEST.read_text())
            manifest['input_sha256'][manifest['fasta']] = '0'*64
            path = Path(tmp)/'manifest.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                validate_inputs(ROOT,path)

    def test_predictions_align_by_id_and_sequence(self):
        records=[{'id':'a','sequence':'AU'},{'id':'b','sequence':'CG'}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.csv'
            path.write_text('variant_id,sequence,fresh_score\nb,CG,2\na,AU,1\n')
            self.assertEqual(align_fresh(records,path),[1.,2.])
            for data in ['a,AU,1\na,AU,2\n','a,AU,1\n','a,AC,1\nb,CG,2\n','a,AU,nan\nb,CG,2\n']:
                path.write_text('variant_id,sequence,fresh_score\n'+data)
                with self.assertRaises(ValueError): align_fresh(records,path)
