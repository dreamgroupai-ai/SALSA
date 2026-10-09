# SALSA decision API: Jev-style typed decisions

A local HTTP service that answers **typed decision questions** about a piece of state.
Each request asks one of three question types: yes/no (`noul`), one of named options
(`choice`), or a level on an ordered scale (`score`). The answer carries a probability
for **every** allowed option.

The model reads the answer in one forward pass from the logits of one token per
option. It never writes free text, so there is nothing to parse.

The wire format is TypeSafe's `POST /v1/systemone`, the interface Jev itself is served
behind. A client written for Jev works against this service by changing its base URL.
How it compares with Jev on JevBench is in [JEVBENCH_RESULTS.md](JEVBENCH_RESULTS.md); training it on your own
decisions is in [JEV_TRAINING.md](JEV_TRAINING.md).

---

## 1. Start it

```bash
./run_jev.sh setup                 # once: isolated vLLM venv + JevBench checkout
./run_jev.sh gpu-check             # refuses GPUs another process is using (FORCE=1 overrides)
./run_jev.sh serve                 # GPU: vLLM + gateway, foreground; Ctrl-C stops both
BACKEND=cpu ./run_jev.sh serve     # CPU: the same API, the same model, much slower
```

The decision endpoint is `http://127.0.0.1:8000/v1/systemone` (`PORT` changes it). Check readiness with:

```bash
curl -s localhost:8000/healthz      # {"status": "ok"} once the model is loaded
curl -s localhost:8000/v1/models    # model name, prompt version, temperature
```

Useful knobs:

| knob | effect |
|---|---|
| `MODEL` | which checkpoint to serve (default `Qwen/Qwen3.8-27B`) |
| `LORA` | serve a SALSA decision adapter from `train_jev.py` (vLLM; see [JEV_TRAINING.md](JEV_TRAINING.md)) |
| `TEMPERATURE` | softmax temperature over the option logits (default 1.0, see §5) |
| `GPUS`, `TP`, `DP` | GPU placement |

`run_jev.sh` binds to localhost only. To reach the service from another machine, use
an SSH tunnel. Alternatively, start the gateway with
`python -m src.jev.server --host 0.0.0.0 --api-key-env MY_TOKEN_VAR`, and every request
must then carry `Authorization: Bearer <token>`.

---

## 2. Ask

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Do not cancel my membership. Please return the duplicate charge.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Select the primary requested action. A mention without a request does not establish intent.",
      "criteria": {
        "cancel": "End an existing subscription",
        "refund": "Return money already charged",
        "other":  "None of these actions is requested"
      }
    }
  }
}'
```

```json
{"model": "salsa-qwen3.8-27b",
 "answers": {"intent": {"type": "choice", "choice": "refund",
                        "probabilities": {"cancel": 0.004, "refund": 0.991, "other": 0.005}}},
 "usage": {"input_tokens": 171, "output_tokens": 1},
 "runtime": {"prompt_version": "salsa-jev-v1", "class_mass": {"intent": 0.999},
             "timing_ms": {"render": 0.8, "engine": 1203.4, "total": 1205.1}}}
```

(The probabilities above are illustrative.)

The three question types:

| `type` | `criteria` | answer fields |
|---|---|---|
| `noul` | `{"true": "...", "false": "..."}`, or omit it | `"noul": P(yes)`; `"probabilities": {"yes", "no"}` |
| `choice` | `{"option_key": "description", ...}`, 2–26 options | `"choice": key`; `"probabilities"` over the keys |
| `score` | `["level 0 description", "level 1 ...", ...]`, 2–10 levels, low → high | `"score": level`; `"probabilities"` over `"0"`…`"n-1"` |

* `state` is any JSON: a string, or an object such as a case file or ticket. Objects are
  shown to the model as indented JSON.
* `questions` may hold several named questions about the same state. They run
  concurrently, and each gets its own answer.
* `model` in the request is accepted and ignored; the server answers with the model it serves.
* Long input is never truncated. A prompt over the context window (32,768 tokens by
  default) is refused with 422.

From Python or a shell, the skill's client saves writing JSON by hand:

```bash
python .claude/skills/salsa-decision-api/decide.py --type noul \
  --instructions "Under the stated policy, is the refund permitted? Treat unproved conditions as not satisfied." \
  --criteria '{"true": "Every required condition is established", "false": "A condition is missing"}' \
  --state "Policy: refunds need a receipt and purchase within 30 days. Bought 12 days ago, no receipt."
```

---

## 3. Errors

| HTTP | meaning | what to do |
|---|---|---|
| 400 | malformed request: unknown type, empty instructions, fewer than 2 options | fix the request |
| 401 | the server requires a bearer token | send `Authorization: Bearer …` |
| 422 | well-formed but declined: more than 26 options, more than 10 levels, prompt longer than the context | split the question, or shorten the state |
| 502 | the model server answered unusably | check `logs/jev/<run>/vllm.log` |
| 503 | the model server is down or still loading | check `/healthz` and retry |

Errors are JSON: `{"error": {"type": "...", "message": "..."}}`.

---

## 4. Writing questions that decide well

The model sees the state, the instructions, every option key with its description, and
nothing else. Each piece below is load-bearing.

* **Instructions are the rubric.** State the decision rule, including the edge cases:
  "Treat unproved conditions as not satisfied", or "Use the highest fully supported level".
  This is where most accuracy is won or lost.
* **Option keys are visible.** Use meaningful keys (`refund`, not `opt2`), and give every
  option a description that separates it from its neighbours.
* **Include an explicit escape option** when none may apply (`"other": "None of these is
  requested"`). Otherwise the model must pick something.
* **`score` levels go from low to high** and should be mutually exclusive.
* **Put the rules in the state** when the decision depends on a policy, a contract or a
  procedure. The model applies what it reads; it has no other context.
* **Precompute arithmetic and dates.** One forward pass does not reason step by step.
  On JevBench's date and arithmetic family the model scored 4 of 15, against 100% on
  trap, adversarial, ambiguous and routing questions. Pass "days since purchase: 12",
  not two dates.
* **Order matters a little.** Options are presented in the order you send them, as
  letters A, B, C…. Keep the order stable between calls you compare.

---

## 5. Reading the answer

* **Use the probabilities, not only the argmax.** They are the model's own distribution
  over the options, not a number it wrote out.
* **Gate on confidence.** The usual pattern is "accept above a threshold, escalate
  below it": for example, send anything under 0.7 to a person or a larger model.
* **Zero-shot, the model is overconfident on hard cases.** On JevBench's hard items, its
  average confidence was 83% while it was right 72% of the time. Jev was at 69% and 73%.
  Treat 0.8 as "probably", not "four times in five".
  `TEMPERATURE` > 1 flattens the distribution. Fit it on your own labelled examples, never
  on the items you report.
* **`runtime.class_mass`** is the share of the model's next-token probability that landed
  on the allowed answers. It is normally above 0.99. Below ~0.9, the model wanted to
  answer something you did not offer, which usually means the options do not cover the
  case or the question does not fit the state.
* `usage.input_tokens` is the prompt length. Cost and latency both scale with it.

---

## 6. Speed

Measured with Qwen3.8-27B, serial, one decision per request. Times are per decision:

| backend | ~150-token decision | hard (~1,300 tokens avg) | longest (~4,000 tokens) | throughput |
|---|---|---|---|---|
| GPU, vLLM 0.31, one B200 per replica | 0.03 s | 0.04 s median | 0.3 s | 22 decisions/s serial; 273/s at concurrency 64 across 4 replicas |

The CPU backend (`BACKEND=cpu`) is much slower: one decision uses a whole socket. Send it
requests one at a time, or batch several questions about the same state into one request.

---

## 7. Compatibility with Jev's API

The same `/v1/systemone` request and response shapes as TypeSafe's API, as used by
JevBench's `typesafe` adapter. The scorer accepts all 231 public JevBench answers as
strictly valid. Differences:

* There is no model routing; `model` is ignored.
* There is no auth unless the gateway is started with `--api-key-env`.
* There is no reasoning mode. Every answer is one forward pass.
* Limits: at most 26 options and at most 10 score levels.
* Extra fields (`runtime`) are returned; Jev clients ignore them.

Implementation: [src/jev/decision.py](src/jev/decision.py) (prompt, option tokens,
readout) and [src/jev/server.py](src/jev/server.py) (HTTP API). Design notes are in
[JEVBENCH.md](JEVBENCH.md) §2.
