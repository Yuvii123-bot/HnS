import unittest

from scripts.suggest_book_repairs import suggest


class SuggestedRepairsTest(unittest.TestCase):
    def test_suggestion_is_reversible_and_unreviewed(self):
        item = {'id': 'x', 'book': 'b', 'text': 'कस्माात्?', 'flags': ['repeated_matra']}
        result = suggest(item)
        self.assertEqual(result['text'], 'कस्माात्?')
        self.assertEqual(result['proposed_text'], 'कस्मात्?')
        self.assertEqual(result['proposed_edits'], 1)
        self.assertEqual(result['repair_review_status'], 'unreviewed')
        self.assertEqual(item['text'], 'कस्माात्?')

    def test_unflagged_is_still_unreviewed(self):
        result = suggest({'id': 'y', 'book': 'b', 'text': 'गौः तृणं खादति।', 'flags': []})
        self.assertEqual(result['proposed_edits'], 0)
        self.assertEqual(result['repair_review_status'], 'unreviewed')


if __name__ == '__main__':
    unittest.main()
