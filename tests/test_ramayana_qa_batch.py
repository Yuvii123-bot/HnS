import random
import unittest
from sanskrit_llm.data import token_feature_table
from sanskrit_llm.tokenizer import AksharaTokenizer
from scripts.train_ramayana_qa_batch import make_batch, read_batch, ROOT

class BatchTest(unittest.TestCase):
    def test_first_batch_and_mask(self):
        tok = AksharaTokenizer.load(ROOT / 'artifacts/tokenizer.json')
        rows, metadata = read_batch(tok, 0, 100)
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row['id'] for row in metadata}), 100)
        x, y, f, total, loss, _ = make_batch(rows, rows, token_feature_table(tok), tok.pad_id, random.Random(1), 8)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(tuple(f.shape[:2]), tuple(x.shape))
        self.assertGreater(total, loss)
        self.assertTrue((y == -1).any())
    def test_missing_batch_refused(self):
        tok = AksharaTokenizer.load(ROOT / 'artifacts/tokenizer.json')
        with self.assertRaises(ValueError):
            read_batch(tok, 100000, 100)

if __name__ == '__main__':
    unittest.main()
