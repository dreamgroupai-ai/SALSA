"""TypeSafe-compatible ``/v1/systemone`` decision API over a vLLM backend.

    JevBench (``typesafe`` adapter) or any client
      POST /v1/systemone  {"state": ..., "model": ..., "questions": {name: question}}
    this gateway
      renders the SALSA prompt (src/jev/decision.py), tokenizes it, and asks vLLM's
      /v1/completions for one token with ``logprob_token_ids`` = the class tokens
    vLLM returns the raw (full-vocabulary) logprobs of exactly those tokens
      -> softmax over the class tokens -> {"answers": {name: typed answer}, "usage", ...}

Why ``logprob_token_ids`` (in the pinned vLLM 0.31 API) rather than ``allowed_token_ids`` + top-k
logprobs: it returns the requested tokens' logprobs whatever their rank, computed
before any sampling transform, so the model's ``generation_config`` defaults
(top_p, top_k) cannot drop a low-probability class.  It also yields the total
probability mass on valid answers, reported per request as ``class_mass``.

Why a gateway in front of ``vllm serve`` rather than an in-process engine: the
OpenAI-compatible server is vLLM's stable interface across releases and already
handles tensor/data/expert parallelism, LoRA adapters and Prometheus metrics.  The
local hop is about a millisecond and is reported in ``runtime.timing_ms``.

Settings come from JEV_* environment variables so that multi-worker uvicorn,
which re-imports this module, sees them too.  ``python -m src.jev.server --help``.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .decision import (
    LETTERS,
    MAX_SCORE_LEVELS,
    PROMPT_VERSION,
    BadQuestion,
    PromptBuilder,
    UnsupportedQuestion,
    build_answer,
    class_distribution,
    parse_question,
)


@dataclass(frozen=True)
class Settings:
    backend: str
    backend_model: str
    tokenizer: str
    model_name: str
    max_model_len: int
    temperature: float
    api_key_env: str
    timeout_s: float

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        return cls(
            backend=e.get("JEV_BACKEND", "http://127.0.0.1:8100").rstrip("/"),
            backend_model=e.get("JEV_BACKEND_MODEL", "qwen3.8-27b"),
            tokenizer=e["JEV_TOKENIZER"],
            model_name=e.get("JEV_MODEL_NAME", "salsa-qwen3.8-27b"),
            max_model_len=int(e.get("JEV_MAX_MODEL_LEN", "32768")),
            temperature=float(e.get("JEV_TEMPERATURE", "1.0")),
            api_key_env=e.get("JEV_API_KEY_ENV", ""),
            timeout_s=float(e.get("JEV_TIMEOUT_S", "300")),
        )


class BackendError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Engine:
    def __init__(self, s: Settings):
        from transformers import AutoTokenizer

        self.s = s
        self.builder = PromptBuilder(AutoTokenizer.from_pretrained(s.tokenizer))
        # The widest question of each type: every class token, checked once at startup.
        for q in ({"type": "noul", "instructions": "q", "criteria": {"true": "t", "false": "f"}},
                  {"type": "choice", "instructions": "q", "criteria": {f"o{i}": "d" for i in range(len(LETTERS))}},
                  {"type": "score", "instructions": "q", "criteria": ["d"] * MAX_SCORE_LEVELS}):
            self.builder.check_alignment("state", parse_question(q))
        # One thread owns the tokenizer: rendering stays off the event loop without
        # concurrent calls into the fast tokenizer.
        self.render_pool = ThreadPoolExecutor(1, thread_name_prefix="render")
        self.client = httpx.AsyncClient(
            base_url=s.backend, timeout=s.timeout_s,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=512))

    async def decide(self, state, question) -> dict:
        q = parse_question(question)
        t0 = time.perf_counter()
        ids, class_ids = await asyncio.get_running_loop().run_in_executor(
            self.render_pool, self.builder.encode, state, q)
        t1 = time.perf_counter()
        if len(ids) + 1 > self.s.max_model_len:
            raise UnsupportedQuestion(
                f"prompt is {len(ids)} tokens; the context window is {self.s.max_model_len}")
        body = {
            "model": self.s.backend_model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": len(class_ids),
            "logprob_token_ids": class_ids,
            "return_tokens_as_token_ids": True,
        }
        r = await self.client.post("/v1/completions", json=body)
        t2 = time.perf_counter()
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text[:300])
        top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        logprobs = {int(k.split(":", 1)[1]): float(v) for k, v in top.items() if k.startswith("token_id:")}
        probs = class_distribution(logprobs, class_ids, self.s.temperature)
        return {
            "answer": build_answer(q, probs),
            "input_tokens": len(ids),
            "class_mass": math.fsum(math.exp(logprobs[i]) for i in class_ids),
            "render_ms": 1e3 * (t1 - t0),
            "engine_ms": 1e3 * (t2 - t1),
        }

    async def close(self):
        await self.client.aclose()
        self.render_pool.shutdown(wait=False)


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


def create_app() -> FastAPI:
    s = Settings.from_env()
    api_key = os.environ.get(s.api_key_env, "") if s.api_key_env else ""
    if s.api_key_env and not api_key:
        raise SystemExit(f"JEV_API_KEY_ENV names {s.api_key_env}, which is unset or empty")
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state["engine"] = Engine(s)
        yield
        await state["engine"].close()

    app = FastAPI(title="SALSA decision API", lifespan=lifespan)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if api_key and not hmac.compare_digest(
                request.headers.get("authorization", ""), f"Bearer {api_key}"):
            return _error(401, "unauthorized", "missing or wrong bearer token")
        t0 = time.perf_counter()
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "body is not JSON")
        if not isinstance(body, dict) or "state" not in body:
            return _error(400, "bad_request", "body must be an object with 'state' and 'questions'")
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            return _error(400, "bad_request", "'questions' must be a non-empty object")
        engine: Engine = state["engine"]
        try:
            outs = await asyncio.gather(*(engine.decide(body["state"], q) for q in questions.values()))
        except BadQuestion as e:
            return _error(400, "bad_request", str(e))
        except UnsupportedQuestion as e:
            return _error(422, "unsupported", str(e))
        except BackendError as e:
            return _error(503 if e.status == 503 else 502, "backend_error", f"backend HTTP {e.status}: {e}")
        except httpx.HTTPError as e:
            return _error(503, "backend_unavailable", f"{type(e).__name__}: {e}")
        except (KeyError, ValueError, TypeError, IndexError) as e:
            return _error(502, "backend_error", f"unusable backend logprobs: {e}")
        names = list(questions)
        return {
            "model": s.model_name,
            "answers": {n: o["answer"] for n, o in zip(names, outs)},
            "usage": {"input_tokens": sum(o["input_tokens"] for o in outs), "output_tokens": len(outs)},
            "runtime": {
                "engine": "salsa-vllm",
                "prompt_version": PROMPT_VERSION,
                "backend_model": s.backend_model,
                "temperature": s.temperature,
                "class_mass": {n: o["class_mass"] for n, o in zip(names, outs)},
                "timing_ms": {
                    "render": max(o["render_ms"] for o in outs),
                    "engine": max(o["engine_ms"] for o in outs),
                    "total": 1e3 * (time.perf_counter() - t0),
                },
            },
        }

    @app.get("/healthz")
    async def healthz():
        try:
            r = await state["engine"].client.get("/health", timeout=5.0)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        return JSONResponse({"status": "ok" if ok else "backend_unavailable"}, status_code=200 if ok else 503)

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": s.model_name, "backend_model": s.backend_model,
                          "prompt_version": PROMPT_VERSION, "temperature": s.temperature}]}

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="SALSA /v1/systemone gateway over a vLLM server")
    ap.add_argument("--backend", default=os.environ.get("JEV_BACKEND", "http://127.0.0.1:8100"))
    ap.add_argument("--backend-model", default=os.environ.get("JEV_BACKEND_MODEL", "qwen3.8-27b"),
                    help="vLLM served-model name, or a LoRA module name to decide with the adapter")
    ap.add_argument("--tokenizer", default=os.environ.get("JEV_TOKENIZER"), required="JEV_TOKENIZER" not in os.environ)
    ap.add_argument("--model-name", default=os.environ.get("JEV_MODEL_NAME", "salsa-qwen3.8-27b"),
                    help="name reported in every response")
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("JEV_MAX_MODEL_LEN", "32768")))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("JEV_TEMPERATURE", "1.0")),
                    help="softmax temperature over the class logits (1.0 = the model's own)")
    ap.add_argument("--api-key-env", default=os.environ.get("JEV_API_KEY_ENV", ""),
                    help="env var holding a bearer token to require; empty = no auth")
    ap.add_argument("--timeout-s", type=float, default=float(os.environ.get("JEV_TIMEOUT_S", "300")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    os.environ.update({
        "JEV_BACKEND": a.backend, "JEV_BACKEND_MODEL": a.backend_model, "JEV_TOKENIZER": a.tokenizer,
        "JEV_MODEL_NAME": a.model_name, "JEV_MAX_MODEL_LEN": str(a.max_model_len),
        "JEV_TEMPERATURE": str(a.temperature), "JEV_API_KEY_ENV": a.api_key_env,
        "JEV_TIMEOUT_S": str(a.timeout_s),
    })
    import uvicorn

    uvicorn.run("src.jev.server:create_app", factory=True, host=a.host, port=a.port,
                workers=a.workers, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
