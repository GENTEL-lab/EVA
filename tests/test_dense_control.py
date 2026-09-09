"""Checks for the recovered author single-expert dense-control architecture."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
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

    def test_dense_forward_update_and_exact_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.config(path)
            model, tokenizer, cfg = _create_dense_model(path, 'cpu')
            self.assertEqual(cfg.num_experts, 1)
            self.assertEqual(cfg.num_experts_per_tok, 1)
            self.assertEqual(cfg.router_aux_loss_coef, 0.0)
            model.output_token_mask = None
            ids = torch.tensor([[10, 11, 12, 13, 10, 11, 12, 13]])
            before = model.model.embed_tokens.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss = model(input_ids=ids, position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
                         sequence_ids=torch.zeros_like(ids), labels=ids).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.model.embed_tokens.weight))
            saved = path / 'weights.pt'
            torch.save(model.state_dict(), saved)
            loaded, _, _ = _create_dense_model(path, 'cpu')
            loaded.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_rejects_multi_expert_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.config(path, experts=4)
            with self.assertRaisesRegex(ValueError, 'single-expert'):
                _create_dense_model(path, 'cpu')

if __name__ == '__main__':
    unittest.main()
