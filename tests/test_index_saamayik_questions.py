import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import index_saamayik_questions as module


class IndexTests(unittest.TestCase):
    def test_pointers_only_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = 'किम्?\nउत्तरम्।\nकः？\n'.encode()
            (root / 'x.sa').write_bytes(data)
            with patch.object(module, 'FILES', {'x.sa': hashlib.sha256(data).hexdigest()}):
                rows = module.index(root)
                self.assertEqual([r['line'] for r in rows], [1, 3])
                self.assertTrue(all(r['answer'] is None for r in rows))
                self.assertNotIn('किम्', str(rows))
                (root / 'x.sa').write_bytes(data + b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    module.index(root)


if __name__ == '__main__':
    unittest.main()
