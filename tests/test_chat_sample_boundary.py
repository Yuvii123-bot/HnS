"""Chat must not generate across corpus example boundaries."""
import unittest
from unittest.mock import patch

from api import index


class Tokenizer:
    pad_id, bos_id, unk_id, eos_id = 0, 1, 2, 3
    vocab = ["अ", "उत्तरम्", "अन्यत्", "<eos>"]

    def encode(self, prompt):
        return [0]

    def special_id(self, name):
        return 0


class Model:
    max_seq_len = 512

    def sample(self, ids, **kwargs):
        self.kwargs = kwargs
        yield 1, False
        yield 3, True
        yield 2, False


class ChatBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.rt = type('Runtime', (), {'tokenizer': Tokenizer(),
                                      'model': self.model, 'feat_table': None})()

    def test_chat_stops_at_end_of_example_even_if_more_lines_requested(self):
        with patch.object(index, 'runtime', return_value=self.rt):
            events = index.do_chat({'message': 'किम्', 'max_lines': 4})
        self.assertEqual(events[-1]['text'], 'उत्तरम्')
        self.assertEqual(self.model.kwargs['max_lines'], 1)
        self.assertNotIn('\n', [e.get('text') for e in events if e['event'] == 'token'])

    def test_continuation_preserves_multiline_behavior(self):
        with patch.object(index, 'runtime', return_value=self.rt):
            events = index.do_chat({'message': 'किम्', 'mode': 'continuation', 'max_lines': 3})
        self.assertEqual(events[-1]['text'], 'उत्तरम्\nअन्यत्')
        self.assertEqual(self.model.kwargs['max_lines'], 3)


if __name__ == '__main__':
    unittest.main()
