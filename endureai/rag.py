"""Semantic retrieval over versioned excerpts from public FEMA guidance."""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import numpy as np
import tiktoken

from .core import ROOT, ResearchAgent, rank_bm25
from .summarizer import _tokenize

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 80
TOP_K = 5
FNSS_SECTIONS = (
    (15, 17, "Key Considerations in Planning for Shelter Set-Up", ["capacity"]),
    (18, 20, "Evaluating the Shelter and Ensuring Accessibility", ["access", "power"]),
    (21, 23, "Shelter Resources and Support Services", ["access", "power"]),
)


def hazard_tag(hazard):
    words = hazard.casefold()
    for tag, phrases in {
        "winter-storm": ("winter", "snow", "blizzard", "ice storm"),
        "hurricane": ("hurricane", "cyclone", "tropical storm"),
        "flood": ("flood",), "wildfire": ("wildfire",), "tornado": ("tornado",),
    }.items():
        if any(phrase in words for phrase in phrases):
            return tag
    return "all-hazards"


def _load_documents(corpus):
    fnss = json.loads(corpus)
    fnss["id"] = "fnss"
    fnss["sections"] = [
        {"title": title, "hazard": "all-hazards", "topics": topics,
         "pages": [page for page in fnss["pages"] if first <= page["page"] <= last]}
        for first, last, title, topics in FNSS_SECTIONS
    ]
    return fnss


def build_chunks(documents, tokenizer):
    """Make overlapping token windows without losing section and PDF page IDs."""
    for doc in documents:
        for section_number, section in enumerate(doc["sections"], 1):
            starts, page_numbers, paragraphs = [], [], []
            char_count = 0
            for page in section["pages"]:
                body = " ".join(page["text"].split())
                if not body:
                    continue
                starts.append(char_count)
                page_numbers.append(page["page"])
                paragraphs.append(body)
                char_count += len(body) + 1
            body = " ".join(paragraphs)
            tokens = tokenizer.encode_ordinary(body)
            if not tokens:
                continue
            _, offsets = tokenizer.decode_with_offsets(tokens)
            start = 0
            while start < len(tokens):
                remaining = len(tokens) - start
                end = len(tokens) if remaining <= 600 else start + CHUNK_TOKENS
                first_char = offsets[start]
                end_char = offsets[end] if end < len(tokens) else len(body)
                first = page_numbers[bisect_right(starts, first_char) - 1]
                last = page_numbers[bisect_right(starts, end_char - 1) - 1]
                yield {
                    "id": f"{doc['id']}-s{section_number}-t{start}",
                    "text": body[first_char:end_char].strip(),
                    "title": doc["title"], "section": section["title"], "published": doc["published"],
                    "hazard": section["hazard"], "topics": ",".join(section["topics"]),
                    "source": "FEMA", "source_url": doc["source_url"],
                    "url": f"{doc['retrieved_url']}#page={first}",
                    "page": first, "page_end": last, "tokens": end - start,
                }
                if end == len(tokens):
                    break
                start = min(end - OVERLAP_TOKENS, max(start + 1, len(tokens) - CHUNK_TOKENS))


class MiniLMEmbeddings:
    """Pool subpassage vectors so the 256-wordpiece model reads the full chunk."""

    def __init__(self):
        self.model = DefaultEmbeddingFunction()

    def __call__(self, texts):
        segments, lengths = [], []
        for text in texts:
            words = text.split()
            # Allow enough headroom for long words and punctuation in WordPiece.
            parts = [" ".join(words[i:i + 100]) for i in range(0, len(words), 100)]
            lengths.append(len(parts))
            segments.extend(parts)
        vectors = self.model(segments)
        start = 0
        embeddings = []
        for length in lengths:
            vector = np.mean(vectors[start:start + length], axis=0)
            embeddings.append(vector / np.linalg.norm(vector))
            start += length
        return embeddings


class SemanticRetriever:
    name = "chroma-minilm"

    def __init__(self, corpus=None, *, embed=None):
        path = Path(corpus) if corpus else ROOT / "data/corpus.json"
        fnss_bytes = path.read_bytes()
        guidance_bytes = (ROOT / "data/fema_guidance.json").read_bytes() if corpus is None else b"[]"
        self.hash = hashlib.sha256(fnss_bytes + guidance_bytes).hexdigest()
        docs = [_load_documents(fnss_bytes), *json.loads(guidance_bytes)]
        self.chunks = tuple(build_chunks(docs, tiktoken.get_encoding("cl100k_base")))
        self.embed = embed if embed is not None else MiniLMEmbeddings()
        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        self.collection = client.create_collection(
            name=f"fema-{uuid4().hex}",
            embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
        )
        if self.chunks:
            texts = [chunk["text"] for chunk in self.chunks]
            self.collection.add(
                ids=[chunk["id"] for chunk in self.chunks], documents=texts, embeddings=self.embed(texts),
                metadatas=[{k: v for k, v in chunk.items() if k not in {"id", "text"}} for chunk in self.chunks],
            )

    def retrieve(self, query, k=TOP_K, *, hazard="all-hazards"):
        if not self.chunks:
            return []
        tags = sorted({"all-hazards", hazard_tag(hazard)})
        response = self.collection.query(
            query_embeddings=self.embed([query]), n_results=k,
            where={"hazard": {"$in": tags}},
            include=["metadatas", "documents", "distances"],
        )
        return [
            {"id": identifier, "text": text, **metadata, "distance": distance}
            for identifier, text, metadata, distance in zip(
                response["ids"][0], response["documents"][0],
                response["metadatas"][0], response["distances"][0], strict=True,
            )
        ]

    def run(self, scenario, original):
        return self.retrieve(audit_query(scenario, original), hazard=scenario["hazard"])


def audit_query(scenario, original):
    codes = {finding["code"] for finding in original["findings"]}
    conditions = [
        "emergency shelter capacity resource availability planning assumptions",
        "facility accessibility and functional needs" if "ACCESS_GAP" in codes else "",
        "electrical outage backup power generation shelter" if scenario["power_required"] else "",
        "contingency for a shelter closure" if "UNAVAILABLE_OPEN" in codes else "",
    ]
    return " ".join(filter(None, [scenario["hazard"], *conditions]))


class LexicalRetriever:
    """Use existing BM25 weighting on the original pages or the semantic index's chunks."""

    def __init__(self, corpus=None, *, chunks=None):
        self.baseline = ResearchAgent(corpus)
        self.hash = self.baseline.hash
        self.name = "bm25-pages" if chunks is None else "bm25-matched-chunks"
        self.chunks = chunks
        self.chunk_tokens = [_tokenize(chunk["text"]) for chunk in chunks] if chunks is not None else None

    def retrieve(self, query, k=TOP_K, *, hazard="all-hazards"):
        if self.chunks is None:
            return self.baseline.retrieve(query, k)
        eligible = [(chunk, tokens) for chunk, tokens in zip(self.chunks, self.chunk_tokens, strict=True)
                    if chunk["hazard"] in {"all-hazards", hazard_tag(hazard)}]
        if not eligible:
            return []
        passages, tokens = zip(*eligible)
        return [{**chunk, "score": score} for score, chunk in rank_bm25(passages, tokens, query)[:k]]

    def run(self, scenario, original):
        return self.retrieve(audit_query(scenario, original), hazard=scenario["hazard"])
