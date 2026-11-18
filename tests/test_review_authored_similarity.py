from pathlib import Path
import tempfile
import unittest
from scripts.review_authored_similarity import find_similar


class SimilarityTests(unittest.TestCase):
    def test_flags_similar_questions_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'batch.tsv'
            p.write_text('category\tquestion\tanswer\nknowledge\tरामः कुत्र अस्ति?\tगृहे।\nknowledge\tरामः कुत्र वसति?\tग्रामे।\n', encoding='utf-8')
            result = find_similar([p], threshold=0.7)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['first']['line'], 2)
            self.assertEqual(result[0]['second']['line'], 3)
            self.assertEqual(result[0]['status'], 'unreviewed_may_be_distinct')


if __name__ == '__main__':
    unittest.main()
