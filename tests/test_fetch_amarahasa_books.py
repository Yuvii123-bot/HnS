import unittest
from scripts.fetch_amarahasa_books import LinkParser, source_sentences
from sanskrit_llm.transliterate import iast_to_devanagari


class SourceParserTest(unittest.TestCase):
    def test_sentences_preserve_spacing_and_punctuation(self):
        html = '<p><s-sent title="Why?"><s-w>kasmāt</s-w>?</s-sent></p><p><s-sent><s-w>hariṇī</s-w> <s-w>bālā</s-w>.</s-sent></p>'
        sentences = source_sentences(html)
        self.assertEqual(sentences, ['kasmāt?', 'hariṇī bālā.'])
        self.assertEqual(iast_to_devanagari(sentences[0], om=False), 'कस्मात्?')
        self.assertEqual(iast_to_devanagari(sentences[1], om=False), 'हरिणी बाला.')

    def test_links_are_recorded(self):
        parser = LinkParser()
        parser.feed('<a href="/books/x/1/">one</a>')
        self.assertEqual(parser.links, ['/books/x/1/'])

    def test_unterminated_rejected(self):
        with self.assertRaises(ValueError):
            source_sentences('<s-sent>kasmāt')


if __name__ == '__main__':
    unittest.main()
