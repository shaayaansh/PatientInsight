"""
Dense (embedding) search over the Asclepius notes (one note per patient).

Same interface as bm25_search.BM25Search, so the rest of the system doesn't care which one it uses:
    search(...) -> [(patient_id, score), ...]   ranked over ALL patients, best first
    get_note(patient_id) -> str

How it works
  1. Each note is split into overlapping word chunks (most notes fit in one chunk; the longest
     ~870-word notes would otherwise be cut off at the model's 512-token limit).
  2. Every chunk is encoded once and the vectors are saved to disk (the "index").
  3. A query is encoded the same way; a patient's score = its BEST chunk's similarity to the query.

Models (see MODELS below; each needs its own index):
    medembed-base   abhinand/MedEmbed-base-v0.1           109M, clinical, sentence-transformers   (default)
    medembed-large  abhinand/MedEmbed-large-v0.1          335M, clinical
    bmretriever     BMRetriever/BMRetriever-410M          410M, biomedical, instruction + last-token pooling
    medcpt          ncbi/MedCPT-Query/Article-Encoder     109M, PubMed search logs, separate query/doc encoders
    bge-base        BAAI/bge-base-en-v1.5                 109M, general-domain control

    pip install torch transformers sentence-transformers numpy pandas pyarrow tqdm

CLI:
    python dense_search.py --build --model medembed-base                 # encode all notes (slow, resumable)
    python dense_search.py --model medembed-base "how is DKA treated and when is insulin switched to subcutaneous" -k 5
    python dense_search.py --build --model medembed-base --limit 2000    # quick trial on 2,000 notes

Python:
    from dense_search import DenseSearch
    dense = DenseSearch.load("medembed-base")
    ranked = dense.search("What symptoms and tests lead to a diagnosis of heart failure?")
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DATA_PATH = Path("data/asclepius/asclepius.parquet")
INDEX_ROOT = Path("data/asclepius/dense_index")
SHARD_SIZE = 10_000          # chunks per saved shard -> an interrupted build resumes where it stopped


# =========================================================================== model registry
@dataclass(frozen=True)
class ModelSpec:
    kind: str                         # "st" (sentence-transformers) | "medcpt" | "last_token" (decoder LLM)
    doc_model: str
    query_model: Optional[str] = None # only for models with a separate query encoder (MedCPT)
    query_template: str = "{text}"    # how a query is wrapped before encoding
    doc_template: str = "{text}"      # how a note chunk is wrapped before encoding
    max_doc_tokens: int = 512
    max_query_tokens: int = 128
    normalize: bool = True            # cosine similarity (True) vs raw dot product (MedCPT was trained with dot product)
    batch_size: int = 32


BMR_TASK = "Given a clinical question, retrieve patient discharge summaries that contain information answering it"

MODELS: Dict[str, ModelSpec] = {
    "medembed-base":  ModelSpec("st", "abhinand/MedEmbed-base-v0.1"),
    "medembed-large": ModelSpec("st", "abhinand/MedEmbed-large-v0.1", batch_size=16),
    "bge-base":       ModelSpec("st", "BAAI/bge-base-en-v1.5",
                                query_template="Represent this sentence for searching relevant passages: {text}"),
    # MedCPT: queries max 64 tokens, [CLS] embedding, dot product (per the model card)
    "medcpt":         ModelSpec("medcpt", "ncbi/MedCPT-Article-Encoder", query_model="ncbi/MedCPT-Query-Encoder",
                                max_query_tokens=64, normalize=False),
    # BMRetriever: one-sentence task instruction on queries, fixed prefix on passages, last-token pooling
    "bmretriever":    ModelSpec("last_token", "BMRetriever/BMRetriever-410M",
                                query_template=BMR_TASK + "\nQuery: {text}",
                                doc_template="Represent this passage\npassage: {text}", batch_size=8),
}


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"                      # Apple Silicon GPU
    return "cpu"


# =========================================================================== encoder
class Encoder:
    """Wraps the three model families behind one API: encode_queries(...) / encode_docs(...)."""

    def __init__(self, spec: ModelSpec, device: Optional[str] = None):
        import torch
        self.spec, self.device, self.torch = spec, device or _device(), torch
        if spec.kind == "st":
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(spec.doc_model, device=self.device)
            self.st.max_seq_length = spec.max_doc_tokens
        else:
            from transformers import AutoModel, AutoTokenizer
            self.doc_tok = AutoTokenizer.from_pretrained(spec.doc_model)
            self.doc_mod = AutoModel.from_pretrained(spec.doc_model).to(self.device).eval()
            if spec.query_model:
                self.q_tok = AutoTokenizer.from_pretrained(spec.query_model)
                self.q_mod = AutoModel.from_pretrained(spec.query_model).to(self.device).eval()
            else:
                self.q_tok, self.q_mod = self.doc_tok, self.doc_mod
            if spec.kind == "last_token" and self.doc_tok.pad_token is None:
                self.doc_tok.pad_token = self.doc_tok.eos_token
            self.doc_tok.padding_side = "right"

    # ------------------------------------------------------------------ public
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        texts = [self.spec.query_template.format(text=t) for t in texts]
        return self._encode(texts, is_query=True)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        texts = [self.spec.doc_template.format(text=t) for t in texts]
        return self._encode(texts, is_query=False)

    # ------------------------------------------------------------------ internals
    def _encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        s = self.spec
        if s.kind == "st":
            return self.st.encode(texts, batch_size=s.batch_size, normalize_embeddings=s.normalize,
                                  convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        tok, mod = (self.q_tok, self.q_mod) if is_query else (self.doc_tok, self.doc_mod)
        max_len = s.max_query_tokens if is_query else s.max_doc_tokens
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), s.batch_size):
                batch = texts[i:i + s.batch_size]
                if s.kind == "last_token":
                    # leave room for EOS and append it, so the last token "sees" the whole text
                    enc = tok(batch, max_length=max_len - 1, truncation=True, padding=False)
                    ids = [x + [tok.eos_token_id] for x in enc["input_ids"]]
                    enc = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
                else:
                    enc = tok(batch, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hidden = mod(**enc).last_hidden_state
                if s.kind == "medcpt":
                    emb = hidden[:, 0, :]                                   # [CLS]
                else:                                                       # last non-pad token (right padding)
                    last = enc["attention_mask"].sum(dim=1) - 1
                    emb = hidden[self.torch.arange(hidden.size(0), device=hidden.device), last]
                if s.normalize:
                    emb = self.torch.nn.functional.normalize(emb, dim=-1)
                out.append(emb.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


# =========================================================================== chunking
def chunk_note(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    """Split a note into overlapping word windows. Most notes (~280 words) become a single chunk."""
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = chunk_words - overlap_words
    return [" ".join(words[i:i + chunk_words]) for i in range(0, max(len(words) - overlap_words, 1), step)]


# =========================================================================== search
class DenseSearch:
    def __init__(self, model_name: str, embeddings: np.ndarray, chunk_pid_idx: np.ndarray,
                 patient_ids: np.ndarray, notes: Dict[int, str], encoder: Optional[Encoder] = None):
        self.model_name, self.spec = model_name, MODELS[model_name]
        self.emb = embeddings                    # (n_chunks, dim) float16 on disk -> float32 here
        self.chunk_pid_idx = chunk_pid_idx       # chunk -> position in patient_ids
        self.patient_ids = patient_ids           # position -> patient_id
        self.notes = notes
        self._encoder = encoder                  # loaded lazily on first query

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(self.spec)
        return self._encoder

    # ------------------------------------------------------------------ build / load
    @staticmethod
    def index_dir(model_name: str) -> Path:
        return INDEX_ROOT / model_name

    @classmethod
    def build(cls, model_name: str = "medembed-base", data_path: Path = DATA_PATH, limit: Optional[int] = None,
              chunk_words: int = 300, overlap_words: int = 50) -> "DenseSearch":
        from tqdm import tqdm
        spec, out = MODELS[model_name], cls.index_dir(model_name)
        out.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(data_path, columns=["patient_id", "note"])
        if limit:
            df = df.iloc[:limit]

        chunks, chunk_pid_idx = [], []
        for pos, note in enumerate(df["note"].tolist()):
            for c in chunk_note(note, chunk_words, overlap_words):
                chunks.append(c); chunk_pid_idx.append(pos)
        meta = {"model_name": model_name, "spec": asdict(spec), "n_patients": len(df), "n_chunks": len(chunks),
                "chunk_words": chunk_words, "overlap_words": overlap_words, "limit": limit}
        meta_path = out / "meta.json"
        if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
            raise RuntimeError(f"{out} holds an index built with different settings. Delete it or use another model name.")
        meta_path.write_text(json.dumps(meta, indent=1))
        np.save(out / "patient_ids.npy", df["patient_id"].to_numpy())
        np.save(out / "chunk_pid_idx.npy", np.array(chunk_pid_idx, dtype=np.int32))
        print(f"[dense_search] {model_name}: {len(df):,} notes -> {len(chunks):,} chunks")

        encoder = Encoder(spec)
        print(f"[dense_search] device={encoder.device}")
        n_shards = (len(chunks) + SHARD_SIZE - 1) // SHARD_SIZE
        t0 = time.time()
        for s in tqdm(range(n_shards), desc="shards"):
            path = out / f"emb_{s:04d}.npy"
            if path.exists():                    # resume: this shard is already done
                continue
            part = encoder.encode_docs(chunks[s * SHARD_SIZE:(s + 1) * SHARD_SIZE])
            np.save(path.with_suffix(".tmp.npy"), part.astype(np.float16))
            path.with_suffix(".tmp.npy").replace(path)
        print(f"[dense_search] encoded in {(time.time() - t0) / 60:.1f} min -> {out}")
        return cls.load(model_name, data_path, encoder=encoder)

    @classmethod
    def load(cls, model_name: str = "medembed-base", data_path: Path = DATA_PATH,
             encoder: Optional[Encoder] = None) -> "DenseSearch":
        d = cls.index_dir(model_name)
        meta = json.loads((d / "meta.json").read_text())
        shards = sorted(d.glob("emb_[0-9][0-9][0-9][0-9].npy"))
        emb = np.concatenate([np.load(p) for p in shards]).astype(np.float32)
        if len(emb) != meta["n_chunks"]:
            raise RuntimeError(f"index incomplete: {len(emb):,}/{meta['n_chunks']:,} chunks. Re-run --build to resume.")
        patient_ids = np.load(d / "patient_ids.npy")
        df = pd.read_parquet(data_path, columns=["patient_id", "note"])
        notes = dict(zip(df["patient_id"], df["note"]))
        return cls(model_name, emb, np.load(d / "chunk_pid_idx.npy"), patient_ids, notes, encoder)

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Rank ALL patients by the similarity of their best-matching chunk to `query`.
        `query` is natural language (a question), NOT a keyword list: dense models are trained on
        sentences, and keyword lists usually make them worse.
        Returns [(patient_id, score), ...] sorted by score, descending.
        """
        q = self.encoder.encode_queries([query])[0]
        chunk_scores = self.emb @ q                                       # (n_chunks,)
        scores = np.full(len(self.patient_ids), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.chunk_pid_idx, chunk_scores)           # max over each patient's chunks
        n = len(scores) if top_k is None else min(top_k, len(scores))
        if n < len(scores):
            idx = np.argpartition(-scores, n - 1)[:n]
            order = idx[np.argsort(-scores[idx], kind="stable")]
        else:
            order = np.argsort(-scores, kind="stable")
        return [(int(self.patient_ids[i]), float(scores[i])) for i in order]

    def get_note(self, patient_id: int) -> str:
        return self.notes[int(patient_id)]


# =========================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description="Dense search over Asclepius notes")
    ap.add_argument("query", nargs="?", help="natural-language query")
    ap.add_argument("--model", default="medembed-base", choices=sorted(MODELS))
    ap.add_argument("--build", action="store_true", help="encode all notes (resumes if interrupted)")
    ap.add_argument("--limit", type=int, default=None, help="build on the first N notes only (for a trial)")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    dense = DenseSearch.build(args.model, limit=args.limit) if args.build else DenseSearch.load(args.model)
    if not args.query:
        return
    t0 = time.time()
    ranked = dense.search(args.query, top_k=args.k)
    print(f"\n[{args.model}] {args.query!r}  ({(time.time() - t0) * 1000:.0f} ms)\n")
    for rank, (pid, score) in enumerate(ranked, 1):
        snippet = " ".join(dense.get_note(pid).split())[:140]
        print(f"{rank:>3}. patient {pid:<7} score={score:.4f}  {snippet}...")


if __name__ == "__main__":
    main()
