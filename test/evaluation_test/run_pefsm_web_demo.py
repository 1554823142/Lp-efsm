"""
启动 P-EFSM 可视化前端示例。

运行:
  python test/evaluation_test/run_pefsm_web_demo.py
  python test/evaluation_test/run_pefsm_web_demo.py --port 8010

然后打开浏览器访问:
  http://127.0.0.1:8000
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动 P-EFSM 可视化 Web Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    uvicorn.run(
        "protocol_infer.visualization.webapp:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
