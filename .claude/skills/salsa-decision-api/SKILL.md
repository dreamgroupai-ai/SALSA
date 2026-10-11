---
name: salsa-decision-api
description: Make Jev-style typed decisions (yes/no, one-of-N choice, ordinal score, each with a probability per option) with the repo's local SALSA decision API (TypeSafe-compatible POST /v1/systemone on Qwen3.8-27B). Use when the user wants to classify, route, policy-check or rate something with that API, check whether it is running, start or stop it, call it from code, read its probabilities, or debug its errors. For benchmarking it against Jev, use JEVBENCH.md and run_jev.sh flow instead.
---

# SALSA decision API

A typed decision is: a **state** (text or JSON), plus a **question** of type `noul`
(yes/no), `choice` (one of named options) or `score` (an ordered level). The answer is a
probability for every allowed option, read in one forward pass. Reference for humans:
`DECISION_API.md` at the repo root.

## 1. Is it running?

```bash
curl -s -m 5 localhost:8000/healthz     # {"status":"ok"} = ready; 503 = model still loading or backend down
```

If nothing answers, the server is not running.

## 2. Starting it (only when the user asks)

Never start a GPU server unless the user asked for it in this conversation: GPUs are
often shared, and a 27B model takes most of one. Starting the CPU backend takes
a whole CPU socket, so ask before that too.

```bash
./run_jev.sh gpu-check                        # GPU: refuses busy GPUs; never pass FORCE=1 on your own
MODEL=<dir or HF id> LORA=<adapter> ./run_jev.sh serve   # a model trained with train_jev.py
./run_jev.sh serve                            # GPU (vLLM 0.31)
BACKEND=cpu ./run_jev.sh serve                # CPU: same API, much slower
```

`serve` runs in the foreground until stopped. Start it as a background command, then
wait for `/healthz` with a Monitor `until curl -sf ...` loop. Do not chain sleeps. Logs
go to `logs/jev/serve_<backend>_<time>/` (`vllm.log`, `gateway.log`). Stop it by
stopping that command; the script's trap takes down vLLM and the gateway, process
groups included.

If `.venv-vllm` is missing, `./run_jev.sh setup` creates it. It needs network and takes ~10 min.

## 3. Asking

Prefer the bundled client. It builds the request, prints every option's probability, and
warns on low confidence or low `class_mass`:

```bash
python .claude/skills/salsa-decision-api/decide.py --type choice \
  --instructions "Which intent does the user's message express?" \
  --criteria '{"track_order": "Wants to know where an order is", "refund": "Wants money back", "other": "None of these"}' \
  --state "Where is my package? I ordered it last week."
# --type noul  [--criteria '{"true": "...", "false": "..."}']
# --type score --criteria '["level 0", "level 1", ...]'          (low -> high, 2-10 levels)
# --state-file case.json | --request body.json (several named questions) | --json (raw response)
```

Exit codes: 0 answered; 2 rejected (400 malformed, 401 auth, 422 too many options or prompt
too long); 3 server unreachable or backend failing.

From code, POST `{"state": ..., "questions": {"<name>": {"type", "instructions", "criteria"}}}` to
`/v1/systemone`. The schema, errors and examples are in `DECISION_API.md` §2–3.

## 4. Writing the question

- The **instructions are the decision rule**. Spell out edge cases, e.g. "Treat unproved
  conditions as not satisfied".
- Use **meaningful option keys** with distinguishing descriptions. Add an `other` / `none`
  option when nothing may apply.
- Put the governing **rules or policy into the state**; the model knows nothing else.
- **Precompute dates and arithmetic** into the state. One forward pass cannot reason
  step by step: it scored 4/15 on JevBench's date and arithmetic family, against 100%
  on trap and adversarial items.

## 5. Reporting the answer

- Report the top option **with its probability**, not just the label.
- Zero-shot, the model is **overconfident on hard cases**: 83% average confidence at 72%
  accuracy on JevBench hard items. Present anything under ~0.7 as uncertain, and say so.
- `class_mass < 0.9` means the model wanted to answer outside the offered options.
  Tell the user the options probably do not fit, and suggest revising them rather than
  trusting the argmax.
- Never present a CPU latency as the service's speed. CPU and GPU differ by orders of magnitude.

## 6. Limits

At most 26 options and at most 10 score levels. The context is 32,768 tokens, and
nothing is truncated: oversize input is a 422. There is no reasoning mode. The `model`
field in requests is ignored.
