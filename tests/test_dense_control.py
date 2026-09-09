"""Checks for the recovered author single-expert dense-control architecture."""
import json
import tempfile
import unittest
from pathlib import Path
from training.eval.scripts.dense_model_loader import _create_dense_model

class DenseControlTests(unittest.TestCase):
    def config(self, path, experts=1):
        cfg = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                   num_attention_heads=4, num_key_value_heads=4,
                   max_position_embeddings=128, use_direction_tokens=True, num_experts=experts,
                   num_experts_per_tok=1, moe_implementation='eager',
                   moe_world_size=1, use_cache=False,
                   attention_dropout=0.0, hidden_dropout=0.0, resid_dropout=0.0)
        (path / 'config.json').write_text(json.dumps(cfg))
        return cfg

    def test_rejects_multi_expert_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.config(path, experts=4)
            with self.assertRaisesRegex(ValueError, 'single-expert'):
                _create_dense_model(path, 'cpu')

if __name__ == '__main__':
    unittest.main()
