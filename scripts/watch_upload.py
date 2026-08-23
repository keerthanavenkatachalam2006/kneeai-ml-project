"""Poll an upload job until it terminates, printing each stage as it happens.

A convenience for watching the pipeline from a terminal, and the thing to reach for
when an upload appears stuck: the stage label comes straight from
``run_case_pipeline``, so it says which engine is currently running rather than just
"processing".

Usage::

    python scripts/watch_upload.py <case_id> [--interval 15]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--max-polls", type=int, default=200)
    args = ap.parse_args()

    url = f"{args.base}/api/uploads/{args.case_id}"
    last_stage: str | None = None

    for _ in range(args.max_polls):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                job = json.load(response)
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.reason} — is the case id right?", flush=True)
            sys.exit(1)
        except urllib.error.URLError as exc:
            print(f"cannot reach the API ({exc.reason}); retrying", flush=True)
            time.sleep(args.interval)
            continue

        # Print every poll so the elapsed clock is visibly moving, but flag the
        # transitions, which are the interesting part.
        marker = "  <-- new stage" if job["stage"] != last_stage else ""
        last_stage = job["stage"]
        print(
            f"{job['elapsed_s']:7.0f}s {job['status']:11s} "
            f"{int(job['progress'] * 100):3d}%  {job['stage'][:58]}{marker}",
            flush=True,
        )

        if job["terminal"]:
            print("\n=== TERMINAL ===", flush=True)
            print(f"status : {job['status']}")
            if job["error"]:
                print(f"error  : {job['error']}")
            print(f"outcome: {json.dumps(job['outcome'], indent=2)}")
            if job["warnings"]:
                print(f"\nwarnings ({len(job['warnings'])}):")
                for warning in job["warnings"]:
                    print(f"  - {warning[:170]}")
            return

        time.sleep(args.interval)

    print("gave up waiting", flush=True)


if __name__ == "__main__":
    main()
