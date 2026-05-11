import unittest

from llm_router.cost import CostLedger, MODEL_PRICING, estimate_cost


class EstimateCostTests(unittest.TestCase):
    def test_zero_tokens_zero_cost(self):
        for model in MODEL_PRICING:
            self.assertEqual(estimate_cost(model, 0, 0), 0.0)

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            estimate_cost("not-a-model", 100, 100)

    def test_negative_tokens_raise(self):
        with self.assertRaises(ValueError):
            estimate_cost("claude-haiku-4-5", -1, 0)

    def test_haiku_cheaper_than_opus_for_same_tokens(self):
        haiku_cost = estimate_cost("claude-haiku-4-5", 100_000, 100_000)
        opus_cost = estimate_cost("claude-opus-4-7", 100_000, 100_000)
        self.assertLess(haiku_cost, opus_cost)

    def test_one_million_input_tokens_matches_table(self):
        cost = estimate_cost("claude-haiku-4-5", 1_000_000, 0)
        self.assertAlmostEqual(cost, MODEL_PRICING["claude-haiku-4-5"].input_per_mtok_usd, places=6)


class CostLedgerTests(unittest.TestCase):
    def test_add_accumulates(self):
        ledger = CostLedger()
        ledger.add("claude-haiku-4-5", 1000, 1000)
        ledger.add("claude-haiku-4-5", 2000, 2000)
        self.assertEqual(ledger.total_input_tokens, 3000)
        self.assertEqual(ledger.total_output_tokens, 3000)
        self.assertGreater(ledger.total_cost_usd, 0.0)

    def test_by_model_groups_calls(self):
        ledger = CostLedger()
        ledger.add("claude-haiku-4-5", 100, 100)
        ledger.add("claude-sonnet-4-6", 100, 100)
        ledger.add("claude-haiku-4-5", 100, 100)
        by = ledger.by_model()
        self.assertEqual(by["claude-haiku-4-5"]["calls"], 2)
        self.assertEqual(by["claude-sonnet-4-6"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
