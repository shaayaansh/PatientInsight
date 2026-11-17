"""
Hybrid search: fuse keyword BM25 and dense retrieval with weighted reciprocal rank fusion (RRF).

    BM25 (core + auxiliary keywords) ──┐
                                       ├── top `depth` of each ──▶ weighted RRF ──▶ one ranked list
    dense (natural-language query) ────┘

RRF uses only ranks, never scores, so BM25's unbounded scores and dense cosine scores never have to be
put on the same scale:

    score(p) = sum over lists i of   w_i / (k + rank_i(p))        rank starts at 1

    k       how much the very top ranks dominate. 60 is the classic default; smaller k (10-20)
            makes rank 1 count much more than rank 10.
    w_i     how much each list's vote counts. BM25 is fixed at 1.0; w_dense is set relative to it.
    depth   only each list's top `depth` take part. A patient below it gets no vote from that list.

HybridSearch does not build or store an index: it holds the BM25Search and DenseSearch objects you
already have and fuses what they return. search() returns [(patient_id, fused_score)], best first,
like the other two retrievers, so anything that consumes one of them can consume this.

Usage (from the project root):
    python hybrid_search.py --core "liver abscess" --aux drainage antibiotics \
        --query "What was the timeline of events for liver abscesses?" -k 10
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Sequence, Tuple

Ranked = List[Tuple[int, float]]        # [(patient_id, score)], best first


# =========================================================================== RRF
def rrf(ranked_lists: Sequence[Ranked], weights: Optional[Sequence[float]] = None, k: float = 60,
        depth: Optional[int] = None, top_k: Optional[int] = None) -> Ranked:
    """
    Weighted reciprocal rank fusion of any number of ranked lists.

    ranked_lists  each a [(patient_id, score)] list sorted best first (scores are ignored)
    weights       one weight per list (default: all 1.0)
    k             RRF constant
    depth         only the top `depth` of each list are used (default: all of it)
    top_k         truncate the fused list (default: return everything that got a vote)

    Ties in the fused score are broken by the best rank the patient had in any list, so the
    output order is deterministic.
    """
    weights = [1.0] * len(ranked_lists) if weights is None else list(weights)
    if len(weights) != len(ranked_lists):
        raise ValueError("need exactly one weight per ranked list")
    scores: Dict[int, float] = {}
    best_rank: Dict[int, int] = {}
    for ranked, w in zip(ranked_lists, weights):
        if w == 0:
            continue
        for rank, (pid, _) in enumerate(ranked[:depth] if depth else ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)
            best_rank[pid] = min(best_rank.get(pid, rank), rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]]))
    return fused[:top_k] if top_k else fused


# =========================================================================== HybridSearch
class HybridSearch:
    def __init__(self, bm25, dense, k: float = 60, w_dense: float = 0.5, depth: int = 100,
                 aux_weight: float = 0.5):
        self.bm25 = bm25                # BM25Search
        self.dense = dense              # DenseSearch
        self.k, self.w_dense, self.depth, self.aux_weight = k, w_dense, depth, aux_weight

    @classmethod
    def load(cls, dense_model: str = "medembed-base", **kwargs) -> "HybridSearch":
        from bm25_search import BM25Search
        from dense_search import DenseSearch
        return cls(BM25Search.load(), DenseSearch.load(dense_model), **kwargs)

    def bm25_ranked(self, core_keywords: Sequence[str], auxiliary_keywords: Optional[Sequence[str]] = None,
                    depth: Optional[int] = None) -> Ranked:
        """BM25's top `depth`, keeping only notes that match at least one keyword (score > 0)."""
        depth = depth or self.depth
        return [(p, s) for p, s in self.bm25.search(core_keywords, auxiliary_keywords,
                                                    aux_weight=self.aux_weight, top_k=depth) if s > 0]

    def dense_ranked(self, query: str, depth: Optional[int] = None) -> Ranked:
        return self.dense.search(query, top_k=depth or self.depth)

    def fuse(self, bm25_ranked: Ranked, dense_ranked: Ranked, top_k: Optional[int] = None,
             k: Optional[float] = None, w_dense: Optional[float] = None, depth: Optional[int] = None) -> Ranked:
        """Fuse two already-computed lists. Lets an evaluation search once and try many settings."""
        return rrf([bm25_ranked, dense_ranked], weights=[1.0, self.w_dense if w_dense is None else w_dense],
                   k=self.k if k is None else k, depth=depth or self.depth, top_k=top_k)

    def search(self, core_keywords: Sequence[str], auxiliary_keywords: Optional[Sequence[str]],
               query: str, top_k: Optional[int] = None) -> Ranked:
        """
        core_keywords / auxiliary_keywords  -> BM25 (exactly as the agent's bm25_search tool uses them)
        query                               -> dense (a natural-language question, not keywords)
        Returns [(patient_id, fused_score)], best first.
        """
        return self.fuse(self.bm25_ranked(core_keywords, auxiliary_keywords),
                         self.dense_ranked(query), top_k=top_k)

    def get_note(self, patient_id: int) -> str:
        return self.bm25.get_note(patient_id)


# =========================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description="Hybrid (BM25 + dense, RRF) search over Asclepius notes")
    ap.add_argument("--core", nargs="+", required=True, help="core keywords for BM25")
    ap.add_argument("--aux", nargs="*", default=[], help="auxiliary keywords for BM25")
    ap.add_argument("--query", required=True, help="natural-language question for dense search")
    ap.add_argument("--dense-model", default="medembed-base")
    ap.add_argument("--rrf-k", type=float, default=60)
    ap.add_argument("--w-dense", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    hs = HybridSearch.load(args.dense_model, k=args.rrf_k, w_dense=args.w_dense, depth=args.depth)
    t0 = time.time()
    b, d = hs.bm25_ranked(args.core, args.aux), hs.dense_ranked(args.query)
    fused = hs.fuse(b, d, top_k=args.k)
    ms = (time.time() - t0) * 1000
    b_rank = {p: r for r, (p, _) in enumerate(b, 1)}
    d_rank = {p: r for r, (p, _) in enumerate(d, 1)}
    print(f"\n[hybrid k={args.rrf_k:g} w_dense={args.w_dense:g} depth={args.depth}]  ({ms:.0f} ms)\n")
    for rank, (pid, score) in enumerate(fused, 1):
        snippet = " ".join(hs.get_note(pid).split())[:110]
        print(f"{rank:>3}. patient {pid:<7} rrf={score:.4f}  bm25#{b_rank.get(pid, '-'):<4} "
              f"dense#{d_rank.get(pid, '-'):<4} {snippet}...")


if __name__ == "__main__":
    main()
