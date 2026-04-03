#!/usr/bin/env python3
"""
localai — local AI assistant CLI (phi3:mini)
Commands: ask, summarize, remind  |  Run bare for interactive REPL
Calls Rust sidecar (:8080) for LLM ops; handles reminders locally.
"""

import argparse
import json
import os
import sys
import requests
from pathlib import Path
from datetime import datetime

RUST_URL   = os.getenv("RUST_SIDECAR_URL", "http://localhost:8080")
STATE_DIR  = Path(os.getenv("STATE_DIR", "./state"))
REMINDERS  = STATE_DIR / "reminders.json"
MODEL      = "phi3:mini"


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def stream_generate(prompt: str) -> None:
    """POST to Rust sidecar, stream chunks straight to stdout."""
    try:
        resp = requests.post(
            f"{RUST_URL}/generate",
            json={"model": MODEL, "prompt": prompt},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode(), end="", flush=True)
        print()
    except requests.ConnectionError:
        sys.exit("[error] Rust sidecar unreachable — run: docker-compose up")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ask(args) -> None:
    stream_generate(" ".join(args.prompt))


def cmd_summarize(args) -> None:
    text = Path(args.file).read_text() if args.file else sys.stdin.read()
    stream_generate(f"Summarize the following text concisely:\n\n{text}")


def cmd_remind(args) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    reminders: list = json.loads(REMINDERS.read_text()) if REMINDERS.exists() else []

    if args.list:
        if not reminders:
            print("No reminders.")
        for i, r in enumerate(reminders, 1):
            print(f"{i}. [{r['time']}] {r['text']}")
        return

    if args.text:
        entry = {
            "time": datetime.now().isoformat(timespec="minutes"),
            "text": " ".join(args.text),
        }
        reminders.append(entry)
        REMINDERS.write_text(json.dumps(reminders, indent=2))
        print(f"Saved: {entry['text']}")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def repl() -> None:
    print(f"localai ({MODEL}) — type 'quit' to exit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        stream_generate(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="localai", description="Local AI assistant")
    sub = parser.add_subparsers(dest="cmd")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("prompt", nargs="+")
    p_ask.set_defaults(func=cmd_ask)

    p_sum = sub.add_parser("summarize", help="Summarize stdin or a file")
    p_sum.add_argument("--file", "-f", metavar="PATH")
    p_sum.set_defaults(func=cmd_summarize)

    p_rem = sub.add_parser("remind", help="Add or list reminders")
    p_rem.add_argument("text", nargs="*", help="Reminder text")
    p_rem.add_argument("--list", "-l", action="store_true")
    p_rem.set_defaults(func=cmd_remind)

    args = parser.parse_args()
    if args.cmd is None:
        repl()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
