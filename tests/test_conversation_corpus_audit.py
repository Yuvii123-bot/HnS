import tempfile
import unittest
from pathlib import Path

from scripts.audit_conversation_corpus import audit


class CorpusAuditTest(unittest.TestCase):
    def test_counts_misaligned_answers_and_exact_leaks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            leaked = '<user> सप्ताहे कति दिनानि सन्ति? <assistant> सप्त दिनानि।'
            bad = '<user> सप्ताहे कति दिनानि सन्ति? <assistant> द्वादश मासाः।'
            (root / 'val.txt').write_text(leaked + '\n', encoding='utf-8')
            part = root / 'part_001.txt'
            part.write_text('\n'.join([leaked, bad, 'प्रथमा विभक्तिः।']) + '\n', encoding='utf-8')
            report = audit(root, [part])
            self.assertEqual(report['validation_unique_exact_overlaps'], 1)
            self.assertEqual(report['fixed_time_rows'], {'total': 2, 'correct': 1, 'incorrect': 1})
            self.assertEqual(report['counts']['no_user_tag'], 1)
            self.assertEqual(report['counts']['paired_tags'], 2)


if __name__ == '__main__':
    unittest.main()
