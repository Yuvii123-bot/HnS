from pathlib import Path
import tempfile
import unittest
import torch
from scripts.colab_train_1b import file_hash, latest_checkpoint, save_checkpoint, verify_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = {'metadata': {'additional_nonpadding_positions': 10}, 'weight': torch.tensor([1])}
            path = save_checkpoint(directory, payload)
            self.assertEqual(path.name, 'tokens-000000000010.pt')
            verify_checkpoint(path)
            self.assertEqual(latest_checkpoint(directory), path)
            self.assertEqual(torch.load(path, weights_only=False)['weight'].item(), 1)
            with self.assertRaises(FileExistsError):
                save_checkpoint(directory, payload)

    def test_corrupt_latest_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_checkpoint(directory, {'metadata': {'additional_nonpadding_positions': 10}})
            newest = save_checkpoint(directory, {'metadata': {'additional_nonpadding_positions': 20}})
            with newest.open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                latest_checkpoint(directory)

    def test_missing_sidecar_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'tokens-000000000020.pt'
            p.write_bytes(b'incomplete')
            with self.assertRaisesRegex(RuntimeError, 'no checksum'):
                latest_checkpoint(Path(tmp))


if __name__ == '__main__':
    unittest.main()
