"""Typed decision question -> SALSA prompt, single-token classes, typed answer.

A TypeSafe ``/v1/systemone`` question (the wire format JevBench's ``typesafe``
adapter speaks) is one of

  noul    {"type": "noul",   "instructions": str, "criteria": {"true": str, "false": str} | null}
  choice  {"type": "choice", "instructions": str, "criteria": {option_key: description, ...}}
  score   {"type": "score",  "instructions": str, "criteria": [level_0_description, ...]}

The request carries no separate label list: the answer keys are the criteria keys
(choice), "yes"/"no" (noul) or the level indices "0".."n-1" (score), and the
returned probabilities must cover exactly that set.

SALSA answers in one forward pass: each option is mapped to a single class token,
the prompt ends at the answer anchor, and the decision is the softmax over the
class tokens' logits at the next position.  Class tokens:

  noul    "yes" / "no"
  choice  "A", "B", ... in the order the criteria arrive (at most 26)
  score   "0", "1", ... the level indices themselves (at most 10)

The anchor is "<ANSWER>\\n", not "<ANSWER>": Qwen's pre-tokenizer glues a
punctuation character onto a following letter, so "<ANSWER>A" tokenizes as
[..., ">A"] and a standalone "A" is never the natural next token there.  After a
newline every class token stands alone.  ``PromptBuilder.check_alignment``
asserts this on the real tokenizer instead of assuming it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

PROMPT_VERSION = "salsa-jev-v1"
ANSWER_PREFIX = "<ANSWER>\n"
ANSWER_SUFFIX = "\n</ANSWER>"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# DeepSeek-V4 ships no chat template, only `encoding/encoding_dsv4.py`.  This is its
# chat (non-thinking) mode for a single user turn, checked character for character
# against `encode_messages([...], thinking_mode="chat")`: the closing `</think>` right
# after the assistant tag is how that format switches reasoning off.
DSV4_CHAT = "<｜begin▁of▁sentence｜><｜User｜>{user}<｜Assistant｜></think>"
MAX_SCORE_LEVELS = 10
QUESTION_TYPES = ("noul", "choice", "score")


class BadQuestion(ValueError):
    """Malformed request (HTTP 400)."""


class UnsupportedQuestion(ValueError):
    """Well-formed request this engine declines (HTTP 422): too many options, or a
    prompt longer than the context window.  JevBench scores a 422 as a wrong
    answer rather than an outage, and never truncates."""


@dataclass(frozen=True)
class TypedQuestion:
    qtype: str
    instructions: str
    keys: tuple          # answer keys returned to the caller, in presentation order
    descriptions: tuple  # str | None per key
    class_tokens: tuple  # the single token the model emits for each key


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def parse_question(q) -> TypedQuestion:
    if not isinstance(q, dict):
        raise BadQuestion("question must be an object")
    qtype = q.get("type")
    if qtype not in QUESTION_TYPES:
        raise BadQuestion(f"question type must be one of {QUESTION_TYPES}, got {qtype!r}")
    instructions = q.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise BadQuestion("instructions must be a non-empty string")
    criteria = q.get("criteria")

    if qtype == "noul":
        if criteria is None:
            descriptions = (None, None)
        elif isinstance(criteria, dict) and set(criteria) <= {"true", "false"}:
            descriptions = (criteria.get("true"), criteria.get("false"))
        else:
            raise BadQuestion("noul criteria must be {'true': ..., 'false': ...} or null")
        keys = tokens = ("yes", "no")
    elif qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise BadQuestion("choice criteria must map at least two option keys to descriptions")
        if len(criteria) > len(LETTERS):
            raise UnsupportedQuestion(f"{len(criteria)} options; at most {len(LETTERS)} are supported")
        keys = tuple(str(k) for k in criteria)
        descriptions = tuple(criteria.values())
        tokens = tuple(LETTERS[: len(keys)])
    else:
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise BadQuestion("score criteria must be a list of at least two level descriptions")
        if len(criteria) > MAX_SCORE_LEVELS:
            raise UnsupportedQuestion(f"{len(criteria)} levels; at most {MAX_SCORE_LEVELS} are supported")
        keys = tokens = tuple(str(i) for i in range(len(criteria)))
        descriptions = tuple(criteria)

    descriptions = tuple(None if d is None else _text(d) for d in descriptions)
    return TypedQuestion(qtype, instructions.strip(), keys, descriptions, tokens)


def render_user_message(state, q: TypedQuestion) -> str:
    lines = [
        "You are a decision engine. Answer the question using only the state below "
        "and any rules it contains.",
        "",
        "<STATE>",
        _text(state),
        "</STATE>",
        "",
        f"Question: {q.instructions}",
        "",
    ]
    if q.qtype == "noul":
        lines.append("Answer yes or no.")
        lines += [f"{k}: {d}" for k, d in zip(q.keys, q.descriptions) if d is not None]
        hint = "yes or no"
    elif q.qtype == "choice":
        lines.append("Options:")
        lines += [f"{t}. {k}" + (f": {d}" if d is not None else "")
                  for t, k, d in zip(q.class_tokens, q.keys, q.descriptions)]
        lines += ["", "Answer with the letter of the single best option."]
        hint = "letter"
    else:
        lines.append("Levels:")
        lines += [f"{k}: {d}" for k, d in zip(q.keys, q.descriptions)]
        lines += ["", "Answer with the number of the level."]
        hint = "level number"
    lines += ["", "Reply in exactly this format:", "<ANSWER>", hint, "</ANSWER>"]
    return "\n".join(lines)


class PromptBuilder:
    """Chat-templated SALSA prompts ending at the answer anchor.

    Thinking stays off: the model must commit to a class token in one forward
    pass, so the template renders the empty ``<think></think>`` block, exactly as
    in the repo's other SALSA pipelines.  Tokenizers without a chat template that
    carry DeepSeek-V4's turn tokens get its chat-mode format instead.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        vocab = tokenizer.get_vocab()
        if getattr(tokenizer, "chat_template", None):
            self.chat_format = "template"
        elif "<｜User｜>" in vocab and "<｜Assistant｜>" in vocab:
            self.chat_format = "dsv4"
        else:
            raise ValueError("tokenizer has no chat template and is not DeepSeek-V4")
        self.token_ids = {}
        for t in ("yes", "no", *LETTERS, *(str(i) for i in range(MAX_SCORE_LEVELS))):
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"class token {t!r} is not a single token (got {ids})")
            self.token_ids[t] = ids[0]
        if len(set(self.token_ids.values())) != len(self.token_ids):
            raise ValueError("class tokens are not distinct")

    def prompt_text(self, state, q: TypedQuestion) -> str:
        user = render_user_message(state, q)
        if self.chat_format == "dsv4":
            return DSV4_CHAT.format(user=user) + ANSWER_PREFIX
        gen = self.tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return gen + ANSWER_PREFIX

    def encode(self, state, q: TypedQuestion) -> tuple[list[int], list[int]]:
        """``(prompt_ids, class_token_ids)``; class ids follow ``q.keys`` order."""
        ids = self.tok.encode(self.prompt_text(state, q), add_special_tokens=False)
        return list(ids), [self.token_ids[t] for t in q.class_tokens]

    def check_alignment(self, state, q: TypedQuestion) -> None:
        """Each class token must be the natural next token after the prompt.

        Tokenizing ``prompt + token + suffix`` has to reproduce the prompt's own
        tokens followed by exactly that class token -- otherwise the readout
        scores a token sequence the model never produces (and a fine-tuned
        adapter would have been trained on a different one).
        """
        text = self.prompt_text(state, q)
        prefix, _ = self.encode(state, q)
        for t in q.class_tokens:
            full = self.tok.encode(text + t + ANSWER_SUFFIX, add_special_tokens=False)
            if full[: len(prefix)] != prefix or full[len(prefix)] != self.token_ids[t]:
                got = self.tok.decode(full[len(prefix): len(prefix) + 2])
                raise AssertionError(f"class token {t!r} is not aligned after the anchor (got {got!r})")


def class_distribution(logprobs: dict, class_ids: list[int], temperature: float = 1.0) -> list[float]:
    """Softmax over the class tokens only, in float64.

    ``logprobs`` maps token id -> log-probability for (at least) every class
    token.  Whether the backend normalised them over the whole vocabulary or over
    the allowed tokens only does not matter: the restricted softmax is the same.
    A missing class token is an error, never a zero.
    """
    missing = [i for i in class_ids if i not in logprobs]
    if missing:
        raise KeyError(f"backend returned no logprob for class token ids {missing}")
    z = [float(logprobs[i]) / temperature for i in class_ids]
    m = max(z)
    if not math.isfinite(m):
        raise ValueError("no class token has a finite logprob")
    e = [math.exp(v - m) for v in z]
    s = math.fsum(e)
    return [v / s for v in e]


def build_answer(q: TypedQuestion, probs: list[float]) -> dict:
    """TypeSafe-shaped answer: argmax plus the full distribution over the keys."""
    dist = dict(zip(q.keys, probs))
    best = q.keys[max(range(len(probs)), key=probs.__getitem__)]
    if q.qtype == "noul":
        return {"type": "noul", "noul": dist["yes"], "probabilities": dist}
    if q.qtype == "choice":
        return {"type": "choice", "choice": best, "probabilities": dist}
    return {"type": "score", "score": int(best), "probabilities": dist}
