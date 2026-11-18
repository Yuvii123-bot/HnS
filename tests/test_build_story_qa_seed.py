import unittest
from unittest.mock import patch

from scripts import build_story_qa_seed as seed


class StorySeedTests(unittest.TestCase):
    def test_selected_pair_has_provenance_and_is_quarantined(self):
        question = {'id': 'b:1:1', 'question_iast': 'kim?', 'question_candidate': 'किम्?',
                    'source_url': 'https://example.org', 'source_html_sha256': 'sha',
                    'context': [{'id': 'b:1:2', 'iast': 'uttaram.', 'devanagari_candidate': 'उत्तरम्.'}]}
        with patch.object(seed, 'SELECTED', {'b:1:1'}), patch.object(seed, 'book_questions', return_value=[question]):
            result = seed.build(None)
        self.assertEqual(result[0]['answer_id'], 'b:1:2')
        self.assertEqual(result[0]['review_status'], 'agent_selected_expert_review_pending')
        self.assertEqual(result[0]['usage'], 'draft_only_not_training_or_evaluation')

    def test_rejects_question_as_answer(self):
        question = {'id': 'b:1:1', 'context': [{'id': 'b:1:2', 'iast': 'kim?'}]}
        with patch.object(seed, 'SELECTED', {'b:1:1'}), patch.object(seed, 'book_questions', return_value=[question]):
            with self.assertRaises(ValueError):
                seed.build(None)


if __name__ == '__main__':
    unittest.main()
