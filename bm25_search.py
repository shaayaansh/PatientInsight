"""
BM25 keyword search over the Asclepius notes (one note per patient).

Uses the `bm25s` library (https://github.com/xhluca/bm25s) with English stopwords
and Snowball stemming (PyStemmer). The index is built once and cached on disk.

    pip install bm25s PyStemmer pandas pyarrow

CLI:
    python bm25_search.py --build                                   # build + cache the index (~1 min)
    python bm25_search.py "congestive heart failure" --aux "echocardiogram" "BNP" -k 5

Python:
    from bm25_search import BM25Search
    bm25 = BM25Search.load_or_build()
    ranked = bm25.search(["congestive heart failure"], auxiliary_keywords=["echocardiogram", "BNP"])
    ranked[:3]        # [(patient_id, score), ...] sorted by score, ALL patients
    bm25.get_note(ranked[0][0])
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import bm25s
import numpy as np
import pandas as pd

try:
    import Stemmer  # PyStemmer
    _STEMMER = Stemmer.Stemmer("english")
except ImportError:  # stemming is optional, but recommended
    _STEMMER = None
    print("[bm25_search] PyStemmer not installed: indexing without stemming.")

DATA_PATH = Path("data/asclepius/asclepius.parquet")
INDEX_DIR = Path("data/asclepius/bm25_index")
STOPWORDS = "en"


def _tokenize(texts: List[str], show_progress: bool = False):
    """Same tokenizer for corpus and queries (lowercase, stopwords, stemming)."""
    return bm25s.tokenize(texts, stopwords=STOPWORDS, stemmer=_STEMMER, show_progress=show_progress)


class BM25Search:
    def __init__(self, retriever: bm25s.BM25, patient_ids: np.ndarray, notes: Dict[int, str]):
        self.retriever = retriever
        self.patient_ids = patient_ids          # position in index -> patient_id
        self.notes = notes                      # patient_id -> note text

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, data_path: Path = DATA_PATH, index_dir: Path = INDEX_DIR,
              k1: float = 1.5, b: float = 0.75) -> "BM25Search":
        df = pd.read_parquet(data_path, columns=["patient_id", "note"])
        print(f"[bm25_search] indexing {len(df):,} notes ...")
        t0 = time.time()
        tokens = _tokenize(df["note"].tolist(), show_progress=True)
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(tokens, show_progress=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(index_dir)
        np.save(index_dir / "patient_ids.npy", df["patient_id"].to_numpy())
        json.dump({"data_path": str(data_path), "stemmer": _STEMMER is not None,
                   "stopwords": STOPWORDS, "k1": k1, "b": b},
                  open(index_dir / "meta.json", "w"), indent=1)
        print(f"[bm25_search] done in {time.time() - t0:.0f}s -> {index_dir}")
        return cls(retriever, df["patient_id"].to_numpy(), dict(zip(df["patient_id"], df["note"])))

    @classmethod
    def load(cls, data_path: Path = DATA_PATH, index_dir: Path = INDEX_DIR) -> "BM25Search":
        meta_path = index_dir / "meta.json"
        if meta_path.exists():
            built_with_stemmer = json.load(open(meta_path)).get("stemmer")
            if built_with_stemmer != (_STEMMER is not None):
                raise RuntimeError(
                    f"Index in {index_dir} was built with stemmer={built_with_stemmer}, but PyStemmer is "
                    f"{'missing' if _STEMMER is None else 'installed'} now. Query terms would not match the "
                    f"index. Fix: `pip install PyStemmer` (or rebuild with `python bm25_search.py --build`).")
        retriever = bm25s.BM25.load(index_dir, show_progress=False)
        patient_ids = np.load(index_dir / "patient_ids.npy")
        df = pd.read_parquet(data_path, columns=["patient_id", "note"])
        return cls(retriever, patient_ids, dict(zip(df["patient_id"], df["note"])))

    @classmethod
    def load_or_build(cls, data_path: Path = DATA_PATH, index_dir: Path = INDEX_DIR) -> "BM25Search":
        if (index_dir / "patient_ids.npy").exists():
            return cls.load(data_path, index_dir)
        return cls.build(data_path, index_dir)

    # ------------------------------------------------------------------ search
    def _scores(self, keywords: Sequence[str]) -> np.ndarray:
        """BM25 score of every document for a bag of keywords (multi-word keywords are split)."""
        text = " ".join(k for k in keywords if k and k.strip())
        if not text:
            return np.zeros(len(self.patient_ids), dtype=np.float32)
        query_tokens = _tokenize([text], show_progress=False)
        # Tokenized(ids, vocab) for the query -> map back to strings so they hit the index vocab
        inv_vocab = {v: k for k, v in query_tokens.vocab.items()}
        terms = [inv_vocab[i] for i in query_tokens.ids[0]]
        terms = [t for t in terms if t in self.retriever.vocab_dict]   # drop out-of-vocabulary terms
        if not terms:
            return np.zeros(len(self.patient_ids), dtype=np.float32)
        return self.retriever.get_scores(terms)

    def search(self, core_keywords: Sequence[str], auxiliary_keywords: Optional[Sequence[str]] = None,
               aux_weight: float = 0.5, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Rank ALL patients by:   score = BM25(core_keywords) + aux_weight * BM25(auxiliary_keywords)

        core_keywords:      keywords taken directly from the query (e.g. "congestive heart failure")
        auxiliary_keywords: expansion terms that may help find evidence (e.g. "BNP", "echocardiogram")
        aux_weight:         how much the expansion terms count relative to the core terms (0 = ignore)
        top_k:              None returns every patient; an int truncates the ranked list

        Returns a list of (patient_id, score), sorted by score descending.
        """
        scores = self._scores(core_keywords)
        if auxiliary_keywords and aux_weight > 0:
            scores = scores + aux_weight * self._scores(auxiliary_keywords)
        n = len(scores) if top_k is None else min(top_k, len(scores))
        if n < len(scores):   # partial sort is much faster than a full argsort
            idx = np.argpartition(-scores, n - 1)[:n]
            order = idx[np.argsort(-scores[idx], kind="stable")]
        else:
            order = np.argsort(-scores, kind="stable")
        return [(int(self.patient_ids[i]), float(scores[i])) for i in order]

    def get_note(self, patient_id: int) -> str:
        return self.notes[int(patient_id)]


def main():
    ap = argparse.ArgumentParser(description="BM25 search over Asclepius notes")
    ap.add_argument("keywords", nargs="*", help="core keywords")
    ap.add_argument("--aux", nargs="*", default=[], help="auxiliary / expansion keywords")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--build", action="store_true", help="(re)build the index")
    args = ap.parse_args()

    bm25 = BM25Search.build() if args.build else BM25Search.load_or_build()
    if not args.keywords:
        return
    t0 = time.time()
    ranked = bm25.search(args.keywords, args.aux, aux_weight=args.aux_weight, top_k=args.k)
    print(f"\ncore={args.keywords} aux={args.aux}  ({(time.time() - t0) * 1000:.0f} ms)\n")
    for rank, (pid, score) in enumerate(ranked, 1):
        snippet = " ".join(bm25.get_note(pid).split())[:140]
        print(f"{rank:>3}. patient {pid:<7} score={score:6.2f}  {snippet}...")


if __name__ == "__main__":
    main()
