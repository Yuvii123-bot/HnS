import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_eval_prompt_overlap import check


class PromptOverlapTest(unittest.TestCase):
    def test_matches_and_does_not_claim_other_prompt_is_held_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = root / 'evaluation.jsonl'
            evaluation.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in [
                {'id': 'seen', 'prompt': 'गौः किं खादति?'},
                {'id': 'unknown', 'prompt': 'अत्र कः?'},
            ]), encoding='utf-8')
            corpus = root / 'part_000.txt'
            corpus.write_text('<user> गौः किं खादति? <assistant> तृणम्।\n', encoding='utf-8')
            result = check([evaluation], [corpus])
            self.assertEqual(result['matching_counts'][str(evaluation)], 1)
            self.assertEqual(result['matched_ids'][str(evaluation)], ['seen'])
            self.assertIn('does not establish', result['limitation'])


if __name__ == '__main__':
    unittest.main()
