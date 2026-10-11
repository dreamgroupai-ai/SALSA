#!/usr/bin/env python3
"""Ask the SALSA decision API (TypeSafe-compatible POST /v1/systemone) one or more typed questions.

    decide.py --type choice --instructions "Which intent does the message express?" \
              --criteria '{"refund": "Wants money back", "cancel": "Wants to end a subscription"}' \
              --state "Please return the duplicate charge."
    decide.py --type noul  --instructions "Is the refund permitted under the policy?" --state-file case.txt
    decide.py --type score --instructions "Rate incident impact." \
              --criteria '["cosmetic", "one user impaired", "many users blocked", "data loss"]' --state "..."
    decide.py --request body.json          # a complete /v1/systemone body, several questions allowed

Prints, per question, the answer, every option's probability and `class_mass` (the
share of the model's next-token probability that landed on the allowed answers),
with warnings when either is low.  `--json` prints the raw response instead.
Exit codes: 0 answered, 2 rejected (400/401/422), 3 server or backend unavailable.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

LOW_CONFIDENCE = 0.6
LOW_CLASS_MASS = 0.9


def parse_state(text: str):
    t = text.strip()
    if t[:1] in "{[":
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
    return text


def build_body(a) -> dict:
    if a.request:
        with open(a.request) as fh:
            return json.load(fh)
    if a.state_file:
        with open(a.state_file) as fh:
            state = parse_state(fh.read())
    elif a.state is not None:
        state = parse_state(a.state)
    else:
        sys.exit("need --state, --state-file or --request")
    if not (a.type and a.instructions):
        sys.exit("need --type and --instructions (or --request)")
    q = {"type": a.type, "instructions": a.instructions}
    if a.criteria:
        q["criteria"] = json.loads(a.criteria)
    return {"state": state, "questions": {a.name: q}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("SALSA_DECISION_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--type", choices=("noul", "choice", "score"))
    ap.add_argument("--instructions")
    ap.add_argument("--criteria", help="JSON: {key: description} for choice, {true:, false:} for noul, "
                                       "[level0, level1, ...] for score")
    ap.add_argument("--state", help="text, or inline JSON")
    ap.add_argument("--state-file")
    ap.add_argument("--name", default="decision", help="question name in the request")
    ap.add_argument("--request", help="a complete /v1/systemone request body (JSON file)")
    ap.add_argument("--api-key-env", default="", help="env var holding a bearer token, if the server requires one")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", action="store_true", help="print the raw response")
    a = ap.parse_args(argv)

    body = build_body(a)
    headers = {"Content-Type": "application/json"}
    if a.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[a.api_key_env]}"
    req = urllib.request.Request(a.endpoint.rstrip("/") + "/v1/systemone", json.dumps(body).encode(),
                                 headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        print(f"HTTP {e.code}: {detail}", file=sys.stderr)
        return 2 if e.code in (400, 401, 422) else 3
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"decision API unreachable at {a.endpoint}: {e}", file=sys.stderr)
        return 3
    latency = time.perf_counter() - t0

    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    mass = (out.get("runtime") or {}).get("class_mass") or {}
    for name, ans in out["answers"].items():
        probs = ans["probabilities"]
        best = max(probs, key=probs.get)
        print(f"{name}: {best}  (p={probs[best]:.3f}, type {ans['type']})")
        for k, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            print(f"    {p:7.4f}  {k}")
        if name in mass:
            print(f"    class_mass {mass[name]:.3f}")
        if probs[best] < LOW_CONFIDENCE:
            print(f"    WARNING: top probability below {LOW_CONFIDENCE} -- treat as uncertain")
        if name in mass and mass[name] < LOW_CLASS_MASS:
            print(f"    WARNING: only {mass[name]:.0%} of the model's probability fell on the allowed "
                  "answers -- the question or options may not fit the state")
    usage = out.get("usage", {})
    print(f"model {out.get('model')}, {usage.get('input_tokens')} input tokens, {latency:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
