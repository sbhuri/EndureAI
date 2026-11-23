import unittest

from endureai.rag import SemanticRetriever


def fixture_embeddings(texts):
    # Exercises real Chroma storage/query without loading a model in unit tests.
    return [[1.0, float(text.casefold().count("shelter") + 1)] for text in texts]


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = SemanticRetriever(embed=fixture_embeddings)

    def test_chunk_lengths_overlap_and_provenance(self):
        chunks = self.rag.chunks
        self.assertEqual({chunk["id"].split("-")[0] for chunk in chunks}, {"fnss", "cpg101", "evacuation"})
        self.assertTrue(all(400 <= x["tokens"] <= 600 for x in chunks))
        self.assertTrue(all(x["page_end"] >= x["page"] and x["section"] for x in chunks))
        groups = {}
        for chunk in chunks:
            group = chunk["id"].rsplit("-t", 1)[0]
            if group in groups:
                self.assertIn(groups[group]["text"].split()[-12], chunk["text"].split())
            groups[group] = chunk

    def test_real_chroma_top_five_and_hazard_filter(self):
        rows = self.rag.retrieve("evacuation", hazard="winter storm")
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["hazard"] for row in rows}, {"all-hazards"})
        self.assertTrue(all(x["source"] == "FEMA" and x["url"].endswith(f"#page={x['page']}") for x in rows))


if __name__ == "__main__":
    unittest.main()
