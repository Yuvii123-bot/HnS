"""Regression tests for conflicting generated question/answer supervision."""
import random
import unittest

from scripts.build_curriculum import time_rows


class TimeAlignmentTest(unittest.TestCase):
    def test_fixed_questions_have_only_matching_answers(self):
        expected = {
            "सप्ताहे कति दिनानि सन्ति?": "सप्त दिनानि।",
            "वर्षे कति मासाः सन्ति?": "द्वादश मासाः।",
            "वर्षे कति ऋतवः सन्ति?": "षट् ऋतवः।",
            "एकस्मिन् वासरे कति होराः भवन्ति?": "चतुर्विंशतिः होराः।",
        }
        rows = time_rows(random.Random(1337), 10000)
        found = set()
        for question, answer in rows:
            if question in expected:
                found.add(question)
                self.assertEqual(expected[question], answer)
        self.assertEqual(found, set(expected))


if __name__ == "__main__":
    unittest.main()
