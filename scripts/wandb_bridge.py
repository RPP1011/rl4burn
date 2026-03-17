#!/usr/bin/env python3
"""Bridge JSONL log output from rl4burn to Weights & Biases.

Usage:
    cargo run --example ppo_cartpole --features "ndarray,json-log" 2>&1 | python scripts/wandb_bridge.py
    # or from a file:
    python scripts/wandb_bridge.py < logs.jsonl
    # offline mode (no API key required):
    cargo run --example ppo_cartpole --features "ndarray,json-log" 2>&1 | python scripts/wandb_bridge.py --offline
"""

import json
import os
import sys

import wandb


def main():
    if "--offline" in sys.argv:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(project="rl4burn")

    for line in sys.stdin:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        step = event.get("step")

        if etype == "scalar":
            wandb.log({event["key"]: event["value"]}, step=step)
        elif etype == "scalars":
            data = {}
            for k, v in event["values"].items():
                data[f"{event['key']}/{k}"] = v
            wandb.log(data, step=step)
        elif etype == "text":
            wandb.log({event["key"]: wandb.Html(f"<pre>{event['text']}</pre>")}, step=step)

    wandb.finish()


if __name__ == "__main__":
    main()
