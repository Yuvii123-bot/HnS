import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_qa_pool_long import audit, extract


class AuditTests(unittest.TestCase):
    def test_rejects_malformed_and_multiturn_rows(self):
        self.assertIsNone(extract('<user> किम् <assistant> अ <user> ख <assistant> ग'))
        self.assertIsNone(extract('prefix <user> किम् <assistant> अ'))
        self.assertEqual(extract('<user> किम् <assistant> अ'), ('किम्', 'अ'))

    def test_conflicts_dedup_and_grammar_keyword(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / 'part_000.txt'
            source.write_text('<user> किम् <assistant> अ\n<user> किम् <assistant> अ\n'
                              '<user> किम् <assistant> ब\n<user> विभक्ति का? <assistant> प्रथमा\n'
                              'काव्यम्\n', encoding='utf-8')
            report = p / 'report.json'
            audit([source], p / 'audit.db', report)
            data = json.loads(report.read_text())
            self.assertEqual(data['unique_exact_qa_pairs'], 3)
            self.assertEqual(data['unique_normalised_questions'], 2)
            self.assertEqual(data['questions_with_multiple_distinct_answers'], 1)
            self.assertEqual(data['distinct_pairs_with_grammar_keyword'], 1)
            self.assertEqual(data['counts']['non_single_turn'], 1)
            self.assertEqual(data['training_ready_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
