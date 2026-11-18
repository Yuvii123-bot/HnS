import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_25k_qa_drafts import book_questions, qa_candidates


class DraftTests(unittest.TestCase):
    def test_question_context_stays_in_chapter_and_does_not_invent_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [dict(id='b:1:1', iast='kim?', devanagari_candidate='किम्?',
                         source_url='https://example.org', source_html_sha256='hash'),
                    dict(id='b:1:2', iast='uttaram.', devanagari_candidate='उत्तरम्.'),
                    dict(id='b:2:1', iast='other.', devanagari_candidate='अन्यत्.')]
            data = ''.join(json.dumps(x) + '\n' for x in rows).encode()
            (p / 'b.jsonl').write_bytes(data)
            (p / 'manifest.json').write_text(json.dumps({'books': [{'book': 'b', 'output_sha256': hashlib.sha256(data).hexdigest()}]}))
            output = book_questions(p)
            self.assertEqual(len(output), 1)
            self.assertIsNone(output[0]['answer'])
            self.assertEqual([x['id'] for x in output[0]['context']], ['b:1:1', 'b:1:2'])
            (p / 'b.jsonl').write_bytes(data + b' ')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                book_questions(p)

    def test_duplicate_questions_are_excluded_without_choosing_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'catalogue.jsonl'
            base = {'source': 'sktqa/sanskrit.tsv', 'answer_raw': 'अ; ब', 'choices_raw': '', 'source_id': '1'}
            data = [dict(base, question='कः?'), dict(base, question='कः?', source_id='2'),
                    dict(base, question='किम्?', source_id='3')]
            p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data), encoding='utf-8')
            output, excluded = qa_candidates(p)
            self.assertEqual(excluded, 2)
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0]['answer_alternatives_raw'], 'अ; ब')
            self.assertIn('not_training', output[0]['usage'])


if __name__ == '__main__':
    unittest.main()
