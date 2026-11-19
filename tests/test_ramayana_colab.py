import random
from pathlib import Path
import torch
from scripts.train_ramayana_colab import prepare, make_batch


class TinyTokenizer:
    bos_id, eos_id, unk_id, pad_id = 1, 2, 3, 0
    def encode(self, text):
        return [3] if 'bad' in text else [4] * len(text)


def test_prepare_and_batch(tmp_path):
    for name in ['aranya', 'ayodhya', 'bala', 'kishkinda', 'sundara']:
        (tmp_path / f'{name}kanda_sanskrit.txt').write_text('verse\nverse\nbad\n', encoding='utf-8')
    tok = TinyTokenizer()
    examples, meta = prepare(tmp_path, tok)
    assert meta['books'] == 5 and meta['chunks'] == 1
    assert meta['unknown_token_lines_skipped'] == 1
    assert examples == [[1, 4, 4, 4, 4, 4, 2]]
    x, y, features, count = make_batch(examples, torch.arange(10), 0, random.Random(0), 2)
    assert count == 12 and x.shape == y.shape == features.shape == (2, 6)
    assert y[0, -1].item() == tok.eos_id
