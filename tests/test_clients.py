import unittest

from llm_router.clients import AnthropicClient, MockClient


class MockClientTests(unittest.TestCase):
    def test_records_each_call(self):
        c = MockClient()
        c.complete("claude-haiku-4-5", "hello")
        c.complete("claude-opus-4-7", "world")
        self.assertEqual(len(c.calls), 2)
        self.assertEqual(c.calls[0][0], "claude-haiku-4-5")
        self.assertEqual(c.calls[1][0], "claude-opus-4-7")

    def test_token_counts_are_positive(self):
        c = MockClient()
        result = c.complete("claude-haiku-4-5", "hello world this is a test")
        self.assertGreater(result.input_tokens, 0)
        self.assertGreater(result.output_tokens, 0)

    def test_returns_model_name_in_result(self):
        c = MockClient()
        result = c.complete("claude-sonnet-4-6", "test")
        self.assertEqual(result.model, "claude-sonnet-4-6")


class AnthropicClientConstructionTests(unittest.TestCase):
    def test_missing_api_key_raises(self):
        with self.assertRaises(ValueError):
            AnthropicClient(api_key=None)
        with self.assertRaises(ValueError):
            AnthropicClient(api_key="")

    def test_constructs_with_key(self):
        # We do not actually call the API; just verify the constructor accepts a key.
        c = AnthropicClient(api_key="sk-ant-anything")
        self.assertIsNotNone(c)


if __name__ == "__main__":
    unittest.main()
