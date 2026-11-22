import os
import re
import json
import math
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pymupdf as fitz
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int


class CampusQueryRAG:
    """Hybrid academic PDF RAG engine.

    Pipeline:
    PDF -> PyMuPDF -> page-aware chunks -> SentenceTransformer embeddings
    -> FAISS + BM25 -> normalized hybrid ranking -> grounded FLAN-T5 answer.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        generation_model: str = "google/flan-t5-small",
        alpha: float = 0.65,
    ):
        self.embedding_model_name = embedding_model
        self.generation_model_name = generation_model
        self.alpha = float(alpha)

        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.generation_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(self.generation_model_name)
        self.generator.eval()

        self.chunks: List[Chunk] = []
        self.texts: List[str] = []
        self.index = None
        self.bm25 = None
        self.embeddings = None
        self.pdf_name = None

    # ------------------------- PDF + chunking -------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\u00ad", "")
        text = re.sub(r"-\n(?=\w)", "", text)          # de-hyphenate line breaks
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _is_noise(text: str) -> bool:
        if not text or len(text.strip()) < 25:
            return True
        letters = sum(c.isalpha() for c in text)
        if letters < 15:
            return True
        return False

    def extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = self.clean_text(page.get_text("text"))
            pages.append((i + 1, text))
        doc.close()
        return pages

    def _chunk_words(self, text: str, page: int, start_id: int, chunk_size: int = 180, overlap: int = 40):
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(1, chunk_size - overlap)
        cid = start_id
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size]).strip()
            if len(piece) >= 40:
                chunks.append(Chunk(piece, page, cid))
                cid += 1
            if start + chunk_size >= len(words):
                break
        return chunks

    def build_index(self, pdf_path: str) -> int:
        if not isinstance(pdf_path, (str, os.PathLike)):
            raise TypeError(f"pdf_path must be a file path, got {type(pdf_path)}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.pdf_name = os.path.basename(pdf_path)
        pages = self.extract_pages(pdf_path)
        nonempty = [(p, t) for p, t in pages if not self._is_noise(t)]

        if not nonempty:
            raise ValueError(
                "No readable text was extracted from this PDF. "
                "It may be scanned/image-only. Version 1 requires a searchable text PDF."
            )

        chunks = []
        cid = 0
        for page, text in nonempty:
            page_chunks = self._chunk_words(text, page, cid)
            chunks.extend(page_chunks)
            cid += len(page_chunks)

        if not chunks:
            raise ValueError("The PDF contains text, but no usable chunks could be created.")

        self.chunks = chunks
        self.texts = [c.text for c in chunks]

        print(f"Extracted {len(self.chunks)} chunks from {len(pages)} pages.")
        print("Creating sentence embeddings...")
        emb = self.embedder.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        self.embeddings = emb

        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

        print("Building BM25 index...")
        tokenized = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        print("Index successfully built.")
        return len(self.chunks)

    # ------------------------- retrieval -------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        lo = float(values.min())
        hi = float(values.max())
        if abs(hi - lo) < 1e-8:
            return np.ones_like(values) * 0.5
        return (values - lo) / (hi - lo)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Index is not built. Upload a PDF and build the index first.")
        if not question.strip():
            return []

        q_emb = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        dense_k = min(max(top_k * 4, 10), len(self.texts))
        dense_scores, dense_ids = self.index.search(q_emb, dense_k)
        dense_scores = dense_scores[0]
        dense_ids = dense_ids[0]

        bm25_raw = np.asarray(self.bm25.get_scores(self._tokenize(question)), dtype=np.float32)
        bm25_norm = self._minmax(bm25_raw)

        # Candidate union: dense top-N + BM25 top-N.
        bm25_ids = np.argsort(bm25_raw)[::-1][:dense_k]
        candidates = set(int(i) for i in dense_ids if i >= 0)
        candidates.update(int(i) for i in bm25_ids if i >= 0)

        dense_map = {int(i): float(s) for i, s in zip(dense_ids, dense_scores) if i >= 0}
        dense_norm_values = self._minmax(np.array(list(dense_map.values()), dtype=np.float32))
        dense_norm_map = {
            idx: float(score) for idx, score in zip(dense_map.keys(), dense_norm_values)
        }

        ranked = []
        for idx in candidates:
            d = dense_norm_map.get(idx, 0.0)
            b = float(bm25_norm[idx])
            hybrid = self.alpha * d + (1.0 - self.alpha) * b
            ranked.append((hybrid, idx, d, b))

        ranked.sort(reverse=True)
        results = []
        for rank, (hybrid, idx, d, b) in enumerate(ranked[:top_k], start=1):
            c = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": c.chunk_id,
                "page": c.page,
                "text": c.text,
                "dense_score": round(d, 4),
                "bm25_score": round(b, 4),
                "hybrid_score": round(float(hybrid), 4),
            })
        return results

    # ------------------------- generation -------------------------
    @staticmethod
    def _clean_answer(answer: str) -> str:
        answer = answer.strip()
        answer = re.sub(r"^(answer|response|output)\s*:\s*", "", answer, flags=re.I)
        answer = answer.strip(" \n\t\"'")
        return answer

    @staticmethod
    def _looks_bad(answer: str, question: str) -> bool:
        a = re.sub(r"[^a-z0-9 ]", " ", answer.lower()).split()
        q = re.sub(r"[^a-z0-9 ]", " ", question.lower()).split()
        if len(a) < 7:
            return True
        if len(set(a)) <= 3:
            return True
        if q and " ".join(q) in " ".join(a):
            return True
        if answer.lower() in {"rag evaluation", "answer", "response"}:
            return True
        return False

    @staticmethod
    def _extractive_fallback(question: str, evidence: List[Dict]) -> str:
        """Simple grounded fallback when the small generator returns a bad answer."""
        if not evidence:
            return "I could not find enough relevant information in the uploaded PDF to answer this question."

        q_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
        scored = []
        for item in evidence:
            sentences = re.split(r"(?<=[.!?])\s+", item["text"])
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 35:
                    continue
                terms = set(re.findall(r"[a-zA-Z0-9]+", sent.lower()))
                overlap = len(q_terms & terms)
                scored.append((overlap, len(sent), sent))
        scored.sort(reverse=True)

        chosen = []
        seen = set()
        for _, _, sent in scored:
            key = sent.lower()
            if key in seen:
                continue
            chosen.append(sent)
            seen.add(key)
            if len(chosen) == 2:
                break
        return " ".join(chosen) if chosen else evidence[0]["text"][:700]

    def generate_answer(self, question: str, evidence: List[Dict]) -> str:
        if not evidence:
            return "I could not find enough relevant information in the uploaded PDF to answer this question."

        # Keep context compact so FLAN-T5-small focuses on the highest-ranked evidence.
        context_parts = []
        for item in evidence[:3]:
            text = item["text"][:900]
            context_parts.append(f"[Page {item['page']}] {text}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are CampusQuery, an academic question-answering assistant.

Answer the question using ONLY the supplied context from the uploaded academic PDF.
Do not use outside knowledge.
Write a direct answer in 2 to 4 complete sentences.
Do not repeat the question.
Do not output only a heading or a short phrase.
If the context does not contain enough information, say that the answer is not available in the provided PDF.

Question:
{question}

Context:
{context}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        output_ids = self.generator.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        answer = self._clean_answer(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))

        if self._looks_bad(answer, question):
            answer = self._extractive_fallback(question, evidence)

        return answer

    def answer(self, question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        evidence = self.retrieve(question, top_k=top_k)
        answer = self.generate_answer(question, evidence)
        return answer, evidence

    # ------------------------- persistence -------------------------
    def save_state(self, folder: str) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty index.")
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "texts": self.texts,
                    "pdf_name": self.pdf_name,
                    "alpha": self.alpha,
                    "embedding_model": self.embedding_model_name,
                    "generation_model": self.generation_model_name,
                },
                f,
            )

    def load_state(self, folder: str) -> None:
        self.index = faiss.read_index(os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.texts = data["texts"]
        self.pdf_name = data.get("pdf_name")
        self.alpha = data.get("alpha", self.alpha)
        self.bm25 = BM25Okapi([self._tokenize(t) for t in self.texts])
