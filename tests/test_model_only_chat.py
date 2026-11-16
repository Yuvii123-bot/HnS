"""Ensure grammar questions cannot bypass model generation."""
import unittest
from unittest.mock import patch

from api import index


class Tokenizer:
    pad_id, bos_id, unk_id, eos_id = 0, 1, 2, 3
    vocab = ["अ"]

    def encode(self, prompt):
        return [0]

    def special_id(self, name):
        return 0


class Model:
    max_seq_len = 512

    def __init__(self):
        self.called = False

    def sample(self, ids, **kwargs):
        self.called = True
        yield 0, False


class ModelOnlyChatTests(unittest.TestCase):
    def test_grammar_question_invokes_model(self):
        model = Model()
        runtime = type("Runtime", (), {"tokenizer": Tokenizer(), "model": model,
                                       "feat_table": None})()
        with patch.object(index, "runtime", return_value=runtime):
            events = index.do_chat({"message": "रामस्य कस्यां विभक्तौ अस्ति?"})
        self.assertTrue(model.called)
        self.assertEqual(events[-1]["text"], "अ")
        self.assertNotEqual(events[-1].get("answerer"), "grammar")

    def test_no_direct_grammar_lookup(self):
        self.assertFalse(hasattr(index, "do_grammar"))
        self.assertFalse(hasattr(index, "grammar_answer"))


if __name__ == "__main__":
    unittest.main()
