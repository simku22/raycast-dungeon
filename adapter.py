"""HTTP adapter — wraps the env in the BenchAnything protocol. Don't edit.

Local dev:  python adapter.py [--port 8765]
The sandbox starts this automatically and assigns a port.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_common.env_sdk import serve

from env import RaycastDungeonEnv

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    print(f"RaycastDungeonEnv adapter → http://{args.host}:{args.port}")
    serve(RaycastDungeonEnv, host=args.host, port=args.port)
