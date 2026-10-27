import numpy as np

from afriembed import metrics as M


def test_recall_and_mrr_identity():
    e = np.eye(5, dtype=np.float32)
    sim = M.sim_matrix(e, e)
    r = M.retrieval_report(sim, np.arange(5))
    assert r["recall@1"] == 1.0 and r["mrr"] == 1.0 and r["n"] == 5 and r["pool"] == 5


def test_ranks_with_ties_and_misses():
    sim = np.array([[0.9, 0.1, 0.5], [0.2, 0.3, 0.9]])
    targets = np.array([0, 1])
    ranks = M.ranks_of_targets(sim, targets)
    assert ranks.tolist() == [1, 2]
    r = M.recall_at_k(sim, targets, ks=(1, 2))
    assert r["recall@1"] == 0.5 and r["recall@2"] == 1.0
    assert abs(M.mrr(sim, targets) - 0.75) < 1e-9


def test_bitext_accuracy_margin():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 8)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = a + 0.01 * rng.normal(size=a.shape).astype(np.float32)
    r = M.bitext_accuracy(M.sim_matrix(a, b))
    assert r["accuracy"] == 1.0 and r["margin_accuracy"] == 1.0 and r["n"] == 20


def test_ndcg_and_recall_qrels():
    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d9": 2}}
    ranked = [["d1", "d3", "d2"], ["d0", "d9"]]
    ndcg = M.ndcg_at_k(ranked, qrels, k=3)
    # q1: dcg = 1 + 1/log2(4) ; idcg = 1 + 1/log2(3)
    q1 = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    q2 = (2 / np.log2(3)) / 2
    assert abs(ndcg - (q1 + q2) / 2) < 1e-9
    assert M.recall_qrels_at_k(ranked, qrels, k=1) == 0.25
    assert M.recall_qrels_at_k(ranked, qrels, k=3) == 1.0


def test_knn_and_clustering_on_separable_data():
    rng = np.random.default_rng(1)
    centers = np.eye(3, dtype=np.float32) * 10
    X = np.vstack([centers[i] + rng.normal(scale=0.1, size=(30, 3)) for i in range(3)]).astype(np.float32)
    y = np.repeat(np.arange(3), 30)
    assert M.knn_accuracy(X, y, k=5) == 1.0
    c = M.clustering_report(X, y)
    assert c["purity"] == 1.0 and c["nmi"] > 0.99 and c["n_clusters"] == 3
