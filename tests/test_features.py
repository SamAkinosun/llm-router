import unittest

from llm_router.features import FEATURE_NAMES, extract_features


def f(prompt):
    return dict(zip(FEATURE_NAMES, extract_features(prompt)))


class FeatureExtractionTests(unittest.TestCase):
    def test_returns_fixed_length_vector(self):
        v = extract_features("hello")
        self.assertEqual(len(v), len(FEATURE_NAMES))
        self.assertTrue(all(isinstance(x, float) for x in v))

    def test_empty_prompt_yields_zeros_for_counts(self):
        v = f("")
        self.assertEqual(v["char_len"], 0.0)
        self.assertEqual(v["word_count"], 0.0)
        self.assertEqual(v["question_count"], 0.0)

    def test_none_prompt_treated_as_empty(self):
        v = f(None)
        self.assertEqual(v["char_len"], 0.0)

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            extract_features(12345)

    def test_question_count_counts_question_marks(self):
        self.assertEqual(f("what? when? why?")["question_count"], 3.0)

    def test_code_block_detected(self):
        v = f("here is code:\n```python\nprint(1)\n```")
        self.assertEqual(v["has_code_block"], 1.0)
        self.assertGreaterEqual(v["code_token_count"], 1.0)

    def test_inline_code_detected(self):
        self.assertEqual(f("call `foo()` here")["has_inline_code"], 1.0)

    def test_url_detected(self):
        self.assertEqual(f("see https://example.com")["has_url"], 1.0)
        self.assertEqual(f("no link here")["has_url"], 0.0)

    def test_math_detected(self):
        self.assertEqual(f("compute $x = 1$")["has_math"], 1.0)
        self.assertEqual(f(r"see \frac{1}{2}")["has_math"], 1.0)

    def test_list_items_counted(self):
        prompt = "things:\n- one\n- two\n- three"
        self.assertEqual(f(prompt)["list_item_count"], 3.0)

    def test_reasoning_tokens_counted(self):
        v = f("Please analyze the trade-offs and design a strategy")
        self.assertGreaterEqual(v["reasoning_token_count"], 3.0)

    def test_uppercase_word_count(self):
        v = f("Use the API and call HTTP from JS")
        self.assertGreaterEqual(v["uppercase_word_count"], 3.0)


if __name__ == "__main__":
    unittest.main()
