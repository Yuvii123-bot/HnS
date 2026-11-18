import unittest

from scripts.prepare_book_review import candidates


class BookReviewTest(unittest.TestCase):
    def test_extracts_multiple_numbered_passages_without_qa_pairing(self):
        text = 'Header\n 1 संजयो बालः।   2 किं संजयः सुखी?\n'
        rows = list(candidates(text, 'book-01'))
        self.assertEqual([row['text'] for row in rows], ['संजयो बालः।', 'किं संजयः सुखी?'])
        self.assertTrue(all(row['review_status'] == 'unreviewed' for row in rows))
        self.assertEqual(rows[1]['source_number'], 2)

    def test_flags_broken_matra_and_incomplete_segment(self):
        rows = list(candidates(' 8 कस्माात्?\n 9 संजयः', 'book-02'))
        self.assertIn('repeated_matra', rows[0]['flags'])
        self.assertIn('possibly_incomplete', rows[1]['flags'])
        self.assertEqual(rows[0]['text'], 'कस्माात्?')


if __name__ == '__main__':
    unittest.main()
