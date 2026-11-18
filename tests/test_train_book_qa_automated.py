import random
import unittest
import torch
from scripts.train_book_qa_automated import batch


class MixedBatchTests(unittest.TestCase):
    def test_example_boundaries_and_mask(self):
        examples = [
            {'kind': 'qa', 'ids': [1, 2, 3, 4, 9], 'loss_start': 1},
            {'kind': 'book', 'ids': [1, 5, 6, 9], 'loss_start': 0},
        ]
        feat = torch.zeros((10, 3))
        x, y, f, positions, loss_positions, counts = batch(examples, feat, 0, random.Random(1), 16)
        self.assertEqual(sum(counts.values()), 16)
        self.assertTrue(counts['qa'] and counts['book'])
        self.assertEqual(f.shape[:2], x.shape)
        self.assertEqual(int((y != -1).sum()), loss_positions)
        self.assertEqual(int((x != 0).sum()), positions)
        for xs, ys in zip(x.tolist(), y.tolist()):
            if xs[1] == 2:
                self.assertEqual(ys[:4], [-1, 3, 4, 9])
            else:
                self.assertEqual(ys[:3], [5, 6, 9])
                self.assertEqual(ys[3], -1)


if __name__ == '__main__':
    unittest.main()
