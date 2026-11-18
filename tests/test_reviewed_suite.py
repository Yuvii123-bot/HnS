import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_reviewed_suite import CATEGORIES, load_suite


def item(category, number):
    return {'id': f'{category}-{number}', 'category': category,
            'prompt': f'प्रश्नः {number}?', 'reference': 'उत्तरम्।',
            'reviewed': True, 'heldout_audited': True, 'audit_evidence': 'audit-record',
            'history': [{'user': 'पूर्वः प्रश्नः?', 'assistant': 'उत्तरम्।'}] if category == 'multiturn' else []}


class ReviewedSuiteTest(unittest.TestCase):
    def write(self, rows, root):
        path = root / 'suite.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        return path

    def test_balanced_suite_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(cat, i) for cat in CATEGORIES for i in range(50)]
            result, digest, counts = load_suite(self.write(rows, root))
            self.assertEqual(len(result), 250)
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(counts[c] == 50 for c in CATEGORIES))

    def test_rejects_insufficient_or_unreviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                load_suite(self.write([item('conversation', 1)], root))
            rows = [item(cat, i) for cat in CATEGORIES for i in range(50)]
            rows[0]['reviewed'] = False
            with self.assertRaises(ValueError):
                load_suite(self.write(rows, root))

    def test_rejects_multiturn_without_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(cat, i) for cat in CATEGORIES for i in range(50)]
            next(r for r in rows if r['category'] == 'multiturn')['history'] = []
            with self.assertRaises(ValueError):
                load_suite(self.write(rows, root))


if __name__ == '__main__':
    unittest.main()
