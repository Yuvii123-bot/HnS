import tempfile
import unittest
from pathlib import Path
from scripts.catalog_qa_sources import SOURCE, catalogue, normalise


class CatalogueTests(unittest.TestCase):
    def test_source_is_quarantined_and_raw_alternatives_preserved(self):
        rows, report = catalogue()
        self.assertEqual(len(rows), 1501)
        self.assertEqual(report['approved_for_training'], 0)
        self.assertTrue(all(x['usage'] == 'quarantine_benchmark_not_training' for x in rows))
        self.assertEqual(rows[0]['answer_raw'], 'नलनीलौ; नीलनलौ')
        self.assertEqual(rows[0]['category'], 'unreviewed')

    def test_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for src in SOURCE.glob('*.tsv'):
                (path / src.name).write_bytes(src.read_bytes())
            with (path / 'sanskrit.tsv').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                catalogue(path)

    def test_normalises_whitespace(self):
        self.assertEqual(normalise('  अ  ब\t'), 'अ ब')


if __name__ == '__main__':
    unittest.main()
