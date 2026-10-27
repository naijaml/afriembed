"""Retrieval, bitext, clustering, and classification metrics on numpy similarity matrices."""

from __future__ import annotations

import numpy as np


def sim_matrix(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) @ d.astype(np.float32).T


def ranks_of_targets(sim: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of the target column for each query row."""
    target_scores = sim[np.arange(sim.shape[0]), targets][:, None]
    return (sim > target_scores).sum(axis=1) + 1


def recall_at_k(sim: np.ndarray, targets: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    r = ranks_of_targets(sim, targets)
    return {f"recall@{k}": float((r <= k).mean()) for k in ks}


def mrr(sim: np.ndarray, targets: np.ndarray) -> float:
    return float((1.0 / ranks_of_targets(sim, targets)).mean())


def retrieval_report(sim: np.ndarray, targets: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    out = recall_at_k(sim, targets, ks)
    out["mrr"] = mrr(sim, targets)
    out["n"] = int(sim.shape[0])
    out["pool"] = int(sim.shape[1])
    return out


def bitext_accuracy(sim: np.ndarray, margin: bool = True) -> dict[str, float]:
    """Aligned pair i<->i. Reports argmax accuracy and (optionally) ratio-margin accuracy a la LASER/mteb."""
    n = sim.shape[0]
    gold = np.arange(n)
    acc = float((sim.argmax(axis=1) == gold).mean())
    out = {"accuracy": acc, "n": n}
    if margin:
        k = min(4, n - 1)
        fwd_nn = np.sort(sim, axis=1)[:, -k:].mean(axis=1, keepdims=True)
        bwd_nn = np.sort(sim, axis=0)[-k:, :].mean(axis=0, keepdims=True)
        scored = sim / ((fwd_nn + bwd_nn) / 2.0)
        out["margin_accuracy"] = float((scored.argmax(axis=1) == gold).mean())
    return out


def ndcg_at_k(ranked_doc_ids: list[list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> float:
    vals = []
    for qid, docs in zip(qrels.keys(), ranked_doc_ids, strict=True):
        rel = qrels[qid]
        gains = [rel.get(d, 0) for d in docs[:k]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def recall_qrels_at_k(
    ranked_doc_ids: list[list[str]], qrels: dict[str, dict[str, int]], k: int = 100
) -> float:
    vals = []
    for qid, docs in zip(qrels.keys(), ranked_doc_ids, strict=True):
        rel = {d for d, r in qrels[qid].items() if r > 0}
        if not rel:
            continue
        vals.append(len(rel & set(docs[:k])) / len(rel))
    return float(np.mean(vals)) if vals else 0.0


def knn_accuracy(emb: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Leave-one-out k-NN accuracy with cosine similarity."""
    sim = sim_matrix(emb, emb)
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k]
    votes = labels[idx]
    pred = np.array([np.bincount(v).argmax() for v in votes])
    return float((pred == labels).mean())


def clustering_report(
    emb: np.ndarray, labels: np.ndarray, n_clusters: int | None = None, seed: int = 0
) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    n_clusters = n_clusters or len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(emb)
    # purity
    contingency = np.zeros((n_clusters, labels.max() + 1), dtype=int)
    for p, l in zip(pred, labels, strict=True):
        contingency[p, l] += 1
    purity = contingency.max(axis=1).sum() / len(labels)
    return {
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "purity": float(purity),
        "n_clusters": int(n_clusters),
    }


def logreg_accuracy(train_x, train_y, test_x, test_y, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(train_x, train_y)
    return float(clf.score(test_x, test_y))
