from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict, Counter
import math

from protocol_infer.evaluation.base import Metric, Evaluator, _safe_div, _entropy, _mean, _l2


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


class ARI(Metric):
    def compute(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        n = len(y_true)
        total_pairs = _comb2(n)
        if n == 0 or total_pairs == 0:
            return {"ari": 0.0}

        ct = Counter(y_true)
        cp = Counter(y_pred)
        contingency = Counter(zip(y_true, y_pred))

        sum_comb = sum(_comb2(v) for v in contingency.values())
        sum_comb_c = sum(_comb2(v) for v in ct.values())
        sum_comb_k = sum(_comb2(v) for v in cp.values())

        expected = (sum_comb_c * sum_comb_k) / total_pairs
        max_index = 0.5 * (sum_comb_c + sum_comb_k)
        denom = max_index - expected
        if denom == 0:
            if len(ct) == 1 and len(cp) == 1:
                return {"ari": 1.0}
            return {"ari": 0.0}
        ari = _safe_div(sum_comb - expected, denom)
        return {"ari": float(ari)}


class NMI(Metric):
    def compute(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        n = len(y_true)
        if n == 0:
            return {"nmi": 0.0}
        ct = Counter(y_true)
        cp = Counter(y_pred)
        joint = Counter(zip(y_true, y_pred))
        mi = 0.0
        for (c, k), v in joint.items():
            pck = v / n
            pc = ct[c] / n
            pk = cp[k] / n
            mi += pck * math.log(_safe_div(pck, pc * pk) + 1e-12)
        ht = _entropy(ct)
        hp = _entropy(cp)
        nmi = _safe_div(2 * mi, ht + hp) if ht + hp > 0 else 0.0
        return {"nmi": nmi}


class HomCompVM(Metric):
    def compute(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        n = len(y_true)
        if n == 0:
            return {"homogeneity": 0.0, "completeness": 0.0, "v_measure": 0.0}
        ct = Counter(y_true)
        cp = Counter(y_pred)
        joint = Counter(zip(y_true, y_pred))
        h_c = _entropy(ct)
        h_k = _entropy(cp)
        cond_h_c = 0.0
        cond_h_k = 0.0
        for k, nk in cp.items():
            sub = {c: joint[(c, k)] for c in ct.keys()}
            cond_h_k += (nk / n) * _entropy(sub)
        for c, nc in ct.items():
            sub = {k: joint[(c, k)] for k in cp.keys()}
            cond_h_c += (nc / n) * _entropy(sub)
        hom = 1.0 - _safe_div(cond_h_k, h_c) if h_c > 0 else 1.0
        comp = 1.0 - _safe_div(cond_h_c, h_k) if h_k > 0 else 1.0
        v = _safe_div(2 * hom * comp, hom + comp) if hom + comp > 0 else 0.0
        return {"homogeneity": hom, "completeness": comp, "v_measure": v}


class Silhouette(Metric):
    def compute(self, features: List[Sequence[float]], labels: List[int]) -> Dict[str, float]:
        n = len(features)
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, k in enumerate(labels):
            clusters[k].append(i)
        if len(clusters) <= 1:
            return {"silhouette": 0.0}
        sil = []
        for i in range(n):
            k = labels[i]
            same = clusters[k]
            if len(same) <= 1:
                sil.append(0.0)
                continue
            a = _mean([_l2(features[i], features[j]) for j in same if j != i])
            b_vals = []
            for k2, idxs in clusters.items():
                if k2 == k:
                    continue
                b_vals.append(_mean([_l2(features[i], features[j]) for j in idxs]))
            b = min(b_vals) if b_vals else 0.0
            s = _safe_div(b - a, max(a, b)) if max(a, b) > 0 else 0.0
            sil.append(s)
        return {"silhouette": _mean(sil)}


class DaviesBouldin(Metric):
    def compute(self, features: List[Sequence[float]], labels: List[int]) -> Dict[str, float]:
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, k in enumerate(labels):
            clusters[k].append(i)
        centroids: Dict[int, List[float]] = {}
        scatters: Dict[int, float] = {}
        for k, idxs in clusters.items():
            if not idxs:
                continue
            dim = len(features[idxs[0]])
            centroid = [0.0] * dim
            for i in idxs:
                for d in range(dim):
                    centroid[d] += features[i][d]
            centroid = [v / len(idxs) for v in centroid]
            centroids[k] = centroid
            scatters[k] = _mean([_l2(features[i], centroid) for i in idxs])
        if len(centroids) <= 1:
            return {"db_index": 0.0}
        R = {}
        keys = list(centroids.keys())
        for i in keys:
            vals = []
            for j in keys:
                if i == j:
                    continue
                m = _l2(centroids[i], centroids[j])
                val = _safe_div(scatters[i] + scatters[j], m) if m > 0 else float("inf")
                vals.append(val)
            R[i] = max(vals) if vals else 0.0
        dbi = _mean(list(R.values()))
        return {"db_index": dbi}


class NoiseRatio(Metric):
    def compute(self, labels: List[int]) -> Dict[str, float]:
        total = len(labels)
        noise = sum(1 for k in labels if k == -1)
        return {"noise_ratio": _safe_div(noise, total) if total > 0 else 0.0}


class ClusteringEvaluator(Evaluator):
    def __init__(self, supervised: bool = True, unsupervised: bool = True):
        metrics: List[Metric] = []
        if supervised:
            metrics += [ARI(), NMI(), HomCompVM()]
        if unsupervised:
            metrics += [Silhouette(), DaviesBouldin(), NoiseRatio()]
        super().__init__(metrics)

    def evaluate(
        self,
        features: Optional[List[Sequence[float]]] = None,
        labels_pred: Optional[List[int]] = None,
        labels_true: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if labels_true is not None and labels_pred is not None:
            for m in [ARI(), NMI(), HomCompVM()]:
                res.update(m.compute(labels_true, labels_pred))
        if features is not None and labels_pred is not None:
            res.update(Silhouette().compute(features, labels_pred))
            res.update(DaviesBouldin().compute(features, labels_pred))
            res.update(NoiseRatio().compute(labels_pred))
        return res
