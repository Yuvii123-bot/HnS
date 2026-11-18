import tempfile
from pathlib import Path
import unittest
import torch
from scripts.train_authored_qa import IsolatedPairs, load_pairs


class Tokenizer:
    pad_id = 0
    eos_id = 9
    def encode(self, text):
        return [1, 2] if '<user>' in text else [3, 4]


class AuthoredTests(unittest.TestCase):
    def test_separate_context_and_answer_loss_mask(self):
        ds = IsolatedPairs([{'question': 'किम्', 'answer': 'अ'}], Tokenizer(), torch.zeros((10, 2)))
        x, y, _ = ds.sample_batch(2)
        self.assertEqual(x.tolist(), [[1, 2, 3, 4], [1, 2, 3, 4]])
        self.assertEqual(y.tolist(), [[-1, 3, 4, 9], [-1, 3, 4, 9]])
        self.assertEqual(ds.positions, 8)
        self.assertEqual(ds.loss_positions, 6)

    def test_duplicate_question_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pairs.tsv'
            p.write_text('category\tquestion\tanswer\na\tकिम्\tअ\na\tकिम्\tब\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_pairs(p)


if __name__ == '__main__':
    unittest.main()
