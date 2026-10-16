import unittest

from rag_engine import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_is_one_chunk(self):
        text = "This is a short document about incident response."
        self.assertEqual(chunk_text(text), [text])

    def test_long_text_chunks_have_content(self):
        text = "word " * 1200
        chunks = chunk_text(text, chunk_size=500, overlap=75)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))
        self.assertLessEqual(max(map(len, chunks)), 500)

    def test_bad_overlap_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
