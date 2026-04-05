"""
对比两种 KMeans 配置在三个工业协议上的效果：
  - baseline : KMeans(n_clusters=8) — 原有固定 K
  - auto_k   : KMeans(n_clusters="auto") — 轮廓系数自动选 K（新方案）

运行方式（从项目根目录）：
    python -m test.evaluation_test.compare_clustering
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from protocol_infer.evaluation.run_evaluation import run_full_evaluation, FullEvalResult
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.algorithm.clustering.kmeans import KMeansClustering

# ---------------------------------------------------------------------------
_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "Data")

_PROTOCOLS: List[Dict] = [
    {"name": "MODBUS", "subdir": "MODBUS"},
    {"name": "DNP3",   "subdir": "DNP3"},
    {"name": "IEC104", "subdir": "IEC60870-104"},
]

# ---------------------------------------------------------------------------
# 带参数注入的评估封装
# ---------------------------------------------------------------------------

def _run(protocol: str, data_dir: str,
         cf_kwargs: dict) -> Tuple[FullEvalResult, Optional[int]]:
    """
    运行完整评估，同时捕获 ControlFlowPipeline 实例以读取 best_k_。
    通过 monkey-patch run_evaluation 模块内的 ControlFlowPipeline 实现注入。
    """
    import protocol_infer.evaluation.run_evaluation as _mod

    captured: List[ControlFlowPipeline] = []   # 用列表在闭包中捕获

    class _PatchedCF(ControlFlowPipeline):
        def __init__(self, **kwargs):
            # cf_kwargs 注入 n_clusters，kwargs 保留调用方传入的其余参数
            merged = {**cf_kwargs, **kwargs}
            super().__init__(**merged)
            captured.append(self)

    orig = _mod.ControlFlowPipeline
    _mod.ControlFlowPipeline = _PatchedCF
    try:
        result = run_full_evaluation(
            protocol=protocol,
            data_dir=data_dir,
            max_pcaps=6,
            max_sessions=200,
        )
    finally:
        _mod.ControlFlowPipeline = orig

    # 读取实际选定的 K（只有 KMeansClustering 才有 best_k_）
    best_k: Optional[int] = None
    if captured:
        clusterer = captured[0].clusterer
        if isinstance(clusterer, KMeansClustering):
            best_k = clusterer.best_k_

    return result, best_k


# ---------------------------------------------------------------------------
# 打印对比表
# ---------------------------------------------------------------------------

_LEGACY_KEYS   = ["precision", "recall", "f1", "accuracy"]
_ENHANCED_KEYS = ["guard_f1", "guard_violation_rate",
                  "action_coverage_jaccard", "state_diff_accuracy"]


def _delta_str(a: float, b: float) -> str:
    d = a - b
    return f"({'+'if d>=0 else ''}{d:.4f})"


def _print_comparison(protocol: str,
                       base: FullEvalResult, auto: FullEvalResult,
                       auto_k: Optional[int],
                       t_base: float, t_auto: float) -> None:
    w = 72
    print(f"\n{'='*w}")
    print(f" {protocol}   train={base.train_sessions}  test={base.test_sessions}")
    print(f"{'─'*w}")
    print(f"  {'Metric':<28} {'Baseline (K=8)':>14}  {'Auto-K (K='+str(auto_k)+')':>14}  {'Δ':>10}")
    print(f"{'─'*w}")

    def row(name: str, d_base: Dict, d_auto: Dict) -> None:
        b = d_base.get(name, float("nan"))
        a = d_auto.get(name, float("nan"))
        print(f"  {name:<28} {b:>14.4f}  {a:>14.4f}  {_delta_str(a,b):>10}")

    for k in _LEGACY_KEYS:
        row(k, base.legacy_efsm, auto.legacy_efsm)

    if base.enhanced_efsm or auto.enhanced_efsm:
        print(f"  {'─'*68}")
        for k in _ENHANCED_KEYS:
            row(k, base.enhanced_efsm, auto.enhanced_efsm)

    print(f"  {'─'*68}")
    print(f"  {'time (s)':<28} {t_base:>14.1f}  {t_auto:>14.1f}  {_delta_str(t_auto,t_base):>10}")
    print(f"  {'K used':<28} {'8':>14}  {str(auto_k) if auto_k else '?':>14}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  Clustering Comparison: Fixed K=8  vs  Auto-K (silhouette score)")
    print("=" * 72)

    for cfg in _PROTOCOLS:
        proto    = cfg["name"]
        data_dir = os.path.join(_DATA_ROOT, cfg["subdir"])
        if not os.path.isdir(data_dir):
            print(f"\n[SKIP] {proto}: 目录不存在 ({data_dir})")
            continue

        # ---------- Baseline: fixed K=8 ----------
        print(f"\n[{proto}] Baseline K=8 ...")
        t0 = time.time()
        try:
            base, _ = _run(proto, data_dir, {"n_clusters": 8, "use_apriori": True})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            continue
        t_base = time.time() - t0
        print(f"  done in {t_base:.1f}s")

        # ---------- Auto-K ----------
        print(f"[{proto}] Auto-K ...")
        t0 = time.time()
        try:
            auto, auto_k = _run(proto, data_dir, {"n_clusters": "auto", "use_apriori": True})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            continue
        t_auto = time.time() - t0
        print(f"  done in {t_auto:.1f}s  (K={auto_k})")

        _print_comparison(proto, base, auto, auto_k, t_base, t_auto)

    print(f"\n{'='*72}\nDone.\n")


if __name__ == "__main__":
    main()
