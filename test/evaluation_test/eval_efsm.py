"""
EFSM 评估入口
=============
对推断的 EFSM 模型运行两类评估并打印清晰指标：

  1. Legacy 评估（基于序列回放）
       - Guard P/R/F1      : guard 函数对正样本/负样本序列的区分能力
       - Action MAE        : action 函数预测值与真实值的平均绝对误差
       - Exact Match Rate  : action 输出逐字段精确匹配率
       - Step Accuracy     : action 预测的增量方向正确率
       - Trace Replay Acc  : 完整会话能被 EFSM 接受的比例
       - FAR / FRR         : 误接受率 / 误拒绝率

  2. Enhanced 评估（与协议规范 GT 对比，仅 MODBUS/DNP3/IEC104）
       - Guard Field P/R/F1       : guard 字段集合覆盖精度
       - Guard Violation Rate     : 推断 guard 被实际流量违反的比例
       - Action Coverage Jaccard  : action 变量集 Jaccard 相似度
       - State Diff Accuracy      : action 预测的变量变化与实际观测吻合率

用法
----
  python test/evaluation_test/eval_efsm.py [--protocol MODBUS] [--data-dir Data/MODBUS]

  不带参数时，自动对 Data/ 下所有可识别协议跑 benchmark。
"""
import argparse
import os
import sys

# 保证在任何工作目录下都能找到项目根
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from protocol_infer.evaluation.run_evaluation import (
    run_full_evaluation,
    run_benchmark,
    FullEvalResult,
)


# ---------------------------------------------------------------------------
# 打印工具
# ---------------------------------------------------------------------------

_SEP = "=" * 72
_SUB = "-" * 72


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_section(title: str, metrics: dict) -> None:
    if not metrics:
        print("  (无指标)")
        return
    max_k = max(len(k) for k in metrics)
    for k, v in sorted(metrics.items()):
        print(f"  {k:<{max_k}}  {_fmt(v)}")


def _print_result(r: FullEvalResult) -> None:
    print(_SEP)
    print(f"  协议  : {r.protocol}")
    print(f"  训练会话 : {r.train_sessions}   测试会话 : {r.test_sessions}")
    print(_SUB)

    print("\n  [Legacy EFSM 指标]  (基于序列回放，不依赖协议规范)")
    _guard = {k: v for k, v in r.legacy_efsm.items() if "guard" in k}
    _action = {k: v for k, v in r.legacy_efsm.items() if "action" in k or "exact" in k or "step" in k}
    _trace = {k: v for k, v in r.legacy_efsm.items() if "trace" in k or "far" in k or "frr" in k}

    if _guard:
        print("\n  >> Guard")
        _print_section("", _guard)
    if _action:
        print("\n  >> Action")
        _print_section("", _action)
    if _trace:
        print("\n  >> Trace / FAR-FRR")
        _print_section("", _trace)

    print("\n  [Enhanced EFSM 指标]  (与协议规范 GT 对比)")
    if r.enhanced_efsm:
        _eg = {k: v for k, v in r.enhanced_efsm.items() if "guard" in k}
        _ea = {k: v for k, v in r.enhanced_efsm.items() if "action" in k}
        _es = {k: v for k, v in r.enhanced_efsm.items() if "state" in k}

        if _eg:
            print("\n  >> Guard（字段级 P/R/F1 + 违规率）")
            _print_section("", _eg)
        if _ea:
            print("\n  >> Action（变量覆盖 Jaccard / P / R）")
            _print_section("", _ea)
        if _es:
            print("\n  >> State Diff（action 预测变量变化准确率）")
            _print_section("", _es)
    else:
        print("\n  (该协议暂无 GT 规范，跳过 Enhanced 评估)")

    print(_SEP)
    print()


def _print_summary(results: list) -> None:
    print("\n" + _SEP)
    print("  汇总（Enhanced 核心指标）")
    print(_SUB)
    header = f"  {'协议':<12}  {'guard_f1':>9}  {'viol_rate':>9}  {'act_jacc':>9}  {'diff_acc':>9}"
    print(header)
    print("  " + "-" * 60)
    for r in results:
        e = r.enhanced_efsm
        gf  = e.get("guard_f1",                 float("nan"))
        vr  = e.get("guard_violation_rate",      float("nan"))
        aj  = e.get("action_coverage_jaccard",   float("nan"))
        da  = e.get("state_diff_accuracy",       float("nan"))
        row = f"  {r.protocol:<12}  {gf:>9.4f}  {vr:>9.4f}  {aj:>9.4f}  {da:>9.4f}"
        print(row)
    print(_SEP + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="对推断 EFSM 模型运行评估并打印清晰指标",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--protocol", "-P", default=None,
                   help="协议名（MODBUS / DNP3 / IEC104）。不填则对所有协议跑 benchmark。")
    p.add_argument("--data-dir", "-d", default=None,
                   help="指定协议的 PCAP 目录（--protocol 存在时使用）。")
    p.add_argument("--data-root", default=os.path.join(_ROOT, "Data"),
                   help="Benchmark 模式下的数据根目录。")
    p.add_argument("--max-pcaps",    type=int,   default=6)
    p.add_argument("--max-sessions", type=int,   default=200)
    p.add_argument("--test-ratio",   type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.protocol:
        # 单协议模式
        proto = args.protocol.upper()
        data_dir = args.data_dir
        if data_dir is None:
            _PROTO_SUBDIR = {
                "MODBUS": "MODBUS", "MODBUSTCP": "MODBUS",
                "IEC104": "IEC60870-104", "IEC60870-104": "IEC60870-104",
                "DNP3": "DNP3",
            }
            sub = _PROTO_SUBDIR.get(proto, proto)
            data_dir = os.path.join(args.data_root, sub)

        print(f"\n正在评估协议: {proto}  数据目录: {data_dir}")
        try:
            result = run_full_evaluation(
                protocol=proto,
                data_dir=data_dir,
                seed=args.seed,
                test_ratio=args.test_ratio,
                max_sessions=args.max_sessions,
                max_pcaps=args.max_pcaps,
            )
            _print_result(result)
        except Exception as exc:
            print(f"[ERROR] 评估失败: {exc}")
            import traceback; traceback.print_exc()

    else:
        # Benchmark 模式：对所有已知协议批量评估
        print(f"\n=== Benchmark 模式  数据根目录: {args.data_root} ===\n")
        results = run_benchmark(
            data_root=args.data_root,
            seed=args.seed,
            test_ratio=args.test_ratio,
            max_sessions=args.max_sessions,
            max_pcaps=args.max_pcaps,
        )
        for r in results:
            _print_result(r)
        if results:
            _print_summary(results)


if __name__ == "__main__":
    main()
