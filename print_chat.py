"""
Pretty-print a full /chat JSON response (no truncation).

Usage (from project root):
  python print_chat.py --message "Your user message here"
  python print_chat.py --url http://127.0.0.1:8000 --message "Hello"

PowerShell-friendly: avoids Invoke-RestMethod table truncation.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser(description="POST /chat and print full JSON")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument("--message", "-m", required=True, help="User message (single turn)")
    args = p.parse_args()

    payload = {"messages": [{"role": "user", "content": args.message}]}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(err_body, file=sys.stderr)
        sys.exit(e.code)
    except urllib.error.URLError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    try:
        obj = json.loads(body)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(body)


if __name__ == "__main__":
    main()
