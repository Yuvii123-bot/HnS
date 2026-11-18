import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_authored_qa import audit


class AuditAuthoredTests(unittest.TestCase):
    def test_detects_cross_batch_conflict_and_eval_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, e = root / 'a.tsv', root / 'b.tsv', root / 'eval.jsonl'
            a.write_text('category\tquestion\tanswer\nknowledge\tकिम्?\tअ\n', encoding='utf-8')
            b.write_text('category\tquestion\tanswer\nknowledge\tकिम्?\tब\n', encoding='utf-8')
            e.write_text(json.dumps({'id': 'q1', 'prompt': 'किम्?'}, ensure_ascii=False) + '\n', encoding='utf-8')
            result = audit([a, b], [e])
            self.assertEqual(result['total_rows'], 2)
            self.assertEqual(result['unique_questions'], 1)
            self.assertEqual(result['duplicate_question_groups'], 1)
            self.assertEqual(result['conflicting_answer_groups'], 1)
            self.assertEqual(result['exact_eval_prompt_matches'][0]['id'], 'q1')
            self.assertNotIn('answer', result['exact_eval_prompt_matches'][0])


if __name__ == '__main__':
    unittest.main()
