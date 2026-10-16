#!/usr/bin/env python3
"""Bounded OpenAI-compatible API client for Task B; no subject-model code."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.example.invalid/v1")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_bytes(canonical(value) + b"\n")
    os.replace(temp, path)


@dataclass(frozen=True)
class ChatResult:
    content: str
    parsed: dict[str, Any] | None
    usage: dict[str, int | None]
    cache_key: str
    cache_hit: bool


class Budget:
    """Reservations remain charged after an attempt; ambiguous failures may have billed."""

    def __init__(self, max_requests: int, max_total_tokens: int,
                 max_usd: float | None = None, input_usd_per_million: float | None = None,
                 output_usd_per_million: float | None = None,
                 state_path: Path | None = None):
        if max_requests < 1 or max_total_tokens < 1:
            raise ValueError("Request and token caps must be positive")
        if max_usd is not None and (input_usd_per_million is None or output_usd_per_million is None):
            raise ValueError("--max-usd needs both documented token rates")
        for name, value in (("max_usd", max_usd), ("input rate", input_usd_per_million),
                            ("output rate", output_usd_per_million)):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self.max_requests = max_requests
        self.max_total_tokens = max_total_tokens
        self.max_usd = max_usd
        self.input_rate = input_usd_per_million
        self.output_rate = output_usd_per_million
        self.requests_reserved = 0
        self.tokens_reserved = 0
        self.usd_reserved = 0.0
        self.lock = threading.Lock()
        self.state_path = state_path
        self.configuration = {"max_requests": max_requests, "max_total_tokens": max_total_tokens,
                              "max_usd": max_usd, "input_rate": input_usd_per_million,
                              "output_rate": output_usd_per_million}

    def _advance(self, state: dict[str, Any], input_bound: int, output_cap: int) -> None:
        charge = input_bound + output_cap
        usd = (input_bound * (self.input_rate or 0) + output_cap * (self.output_rate or 0)) / 1_000_000
        if state["requests_reserved"] + 1 > self.max_requests:
            raise RuntimeError("Hard request cap reached")
        if state["tokens_reserved"] + charge > self.max_total_tokens:
            raise RuntimeError("Hard token reservation cap reached")
        if self.max_usd is not None and state["usd_reserved"] + usd > self.max_usd:
            raise RuntimeError("Hard USD reservation cap reached")
        state["requests_reserved"] += 1
        state["tokens_reserved"] += charge
        state["usd_reserved"] += usd
        for field in ("requests_reserved", "tokens_reserved", "usd_reserved"):
            setattr(self, field, state[field])

    def reserve(self, input_bound: int, output_cap: int) -> None:
        if input_bound < 0 or output_cap < 1:
            raise ValueError("Invalid token reservation")
        with self.lock:
            if self.state_path is None:
                state = {field: getattr(self, field) for field in
                         ("requests_reserved", "tokens_reserved", "usd_reserved")}
                self._advance(state, input_bound, output_cap)
                return
            with file_lock(self.state_path.with_suffix(self.state_path.suffix + ".lock")):
                state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
                    "schema_version": "pie-B-persistent-budget-v1", "configuration": self.configuration,
                    "requests_reserved": 0, "tokens_reserved": 0, "usd_reserved": 0.0}
                if state.get("configuration") != self.configuration:
                    raise RuntimeError("Persistent budget configuration changed; cap cannot reset on resume")
                if (state.get("schema_version") != "pie-B-persistent-budget-v1" or
                        any(type(state.get(k)) is not int or state[k] < 0
                            for k in ("requests_reserved", "tokens_reserved")) or
                        isinstance(state.get("usd_reserved"), bool) or
                        not isinstance(state.get("usd_reserved"), (int, float)) or
                        not math.isfinite(state["usd_reserved"]) or state["usd_reserved"] < 0):
                    raise RuntimeError("Invalid persistent budget state; reservations cannot reset")
                self._advance(state, input_bound, output_cap)
                state["updated_at_utc"] = utc_now()
                atomic_json(self.state_path, state)


@contextmanager
def file_lock(path: Path):
    """Separate lock file survives atomic replacement of the protected JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class APIClient:
    def __init__(self, *, model: str, base_url: str, key_env: str, cache_dir: Path,
                 ledger_path: Path, budget: Budget, timeout_s: float = 25,
                 max_retries: int = 0, max_concurrent: int = 1,
                 reasoning_effort: str | None = None, omit_temperature: bool = False):
        if not model or not key_env:
            raise ValueError("Model and key environment variable are required")
        if not 1 <= max_concurrent <= 16 or not 0 <= max_retries <= 5:
            raise ValueError("Concurrency or retries outside bounded range")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env
        self.cache_dir = cache_dir
        self.ledger_path = ledger_path
        self.budget = budget
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.semaphore = threading.BoundedSemaphore(max_concurrent)
        self.log_lock = threading.Lock()
        self.reasoning_effort = reasoning_effort
        self.omit_temperature = omit_temperature

    def _log(self, entry: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock, self.ledger_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"at_utc": utc_now(), **entry}, sort_keys=True) + "\n")

    def chat(self, messages: list[dict[str, str]], *, prompt_version: str,
             max_output_tokens: int = 128, temperature: float = 0,
             require_json: bool = True, cache_only: bool = False) -> ChatResult:
        payload = self.payload(messages, max_output_tokens=max_output_tokens,
                               temperature=temperature, require_json=require_json)
        cache_key = sha256({"endpoint": self.base_url, "prompt_version": prompt_version,
                            "payload": payload})
        # An identical question used by two methods/processes pays at most once.
        with file_lock(self.cache_dir / (cache_key + ".lock")):
            return self._chat_locked(messages, payload, cache_key, prompt_version,
                                     max_output_tokens, require_json, cache_only)

    def payload(self, messages: list[dict[str, str]], *, max_output_tokens: int,
                temperature: float = 0, require_json: bool = True) -> dict[str, Any]:
        if not messages or max_output_tokens < 1:
            raise ValueError("Nonempty messages and positive output cap required")
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "max_completion_tokens": max_output_tokens}
        if not self.omit_temperature:
            payload["temperature"] = temperature
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if require_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _chat_locked(self, messages: list[dict[str, str]], payload: dict[str, Any], cache_key: str,
                     prompt_version: str, max_output_tokens: int, require_json: bool,
                     cache_only: bool = False) -> ChatResult:
        log_meta = {"endpoint": self.base_url, "model": self.model,
                    "prompt_version": prompt_version,
                    "reasoning_effort": self.reasoning_effort,
                    "temperature": payload.get("temperature"),
                    "input_usd_per_million": self.budget.input_rate,
                    "output_usd_per_million": self.budget.output_rate,
                    "response_mode": "json_object" if require_json else "text"}
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self._log({"cache_key": cache_key, **log_meta, "cache_hit": True,
                       "status": "ok", "usage": saved["usage"]})
            return ChatResult(saved["content"], saved["parsed"], saved["usage"], cache_key, True)
        if cache_only:
            raise RuntimeError("Required cached API reply is unavailable; no request sent")
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(f"{self.key_env} is unavailable to this process; no request sent")
        input_bound = len(canonical(messages)) + 64 * len(messages) + 128
        last_error: Exception | None = None
        with self.semaphore:
            for attempt in range(self.max_retries + 1):
                self.budget.reserve(input_bound, max_output_tokens)
                usage: dict[str, int | None] = {}
                request = urllib.request.Request(
                    self.base_url + "/chat/completions", data=canonical(payload),
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    method="POST")
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                        data = json.loads(response.read())
                    raw_usage = data.get("usage") or {}
                    usage = {k: raw_usage.get(k) for k in
                             ("prompt_tokens", "completion_tokens", "total_tokens")}
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Empty API content")
                    parsed = json.loads(content) if require_json else None
                    if require_json and not isinstance(parsed, dict):
                        raise ValueError("Expected a JSON object")
                    atomic_json(path, {"content": content, "parsed": parsed, "usage": usage,
                                       "model": data.get("model", self.model),
                                       "created_at_utc": utc_now(), "prompt_version": prompt_version})
                    self._log({"cache_key": cache_key, **log_meta,
                               "cache_hit": False, "status": "ok", "attempt": attempt + 1,
                               "usage": usage, "input_token_bound": input_bound,
                               "output_token_cap": max_output_tokens})
                    return ChatResult(content, parsed, usage, cache_key, False)
                except urllib.error.HTTPError as exc:
                    code = exc.code
                    last_error = RuntimeError(f"API HTTP {code}")
                    retry = code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries
                    self._log({"cache_key": cache_key, **log_meta,
                               "cache_hit": False, "status": "http_error",
                               "attempt": attempt + 1, "http_status": code, "retry": retry,
                               "input_token_bound": input_bound, "output_token_cap": max_output_tokens})
                    if not retry:
                        break
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_error = RuntimeError(f"API transport failure: {type(exc).__name__}")
                    retry = attempt < self.max_retries
                    self._log({"cache_key": cache_key, **log_meta,
                               "cache_hit": False, "status": "transport_error",
                               "attempt": attempt + 1, "retry": retry,
                               "input_token_bound": input_bound, "output_token_cap": max_output_tokens})
                    if not retry:
                        break
                except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    last_error = RuntimeError(f"API response format failure: {type(exc).__name__}")
                    self._log({"cache_key": cache_key, **log_meta, "cache_hit": False,
                               "status": "invalid_response", "attempt": attempt + 1,
                               "usage": usage,
                               "input_token_bound": input_bound, "output_token_cap": max_output_tokens})
                    break
                time.sleep(min(8.0, 0.5 * 2 ** attempt) + random.random() * 0.1)
        raise last_error or RuntimeError("API request failed")


def list_models(base_url: str, key_env: str, timeout_s: float) -> list[str]:
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} is unavailable to this process")
    root = base_url.rstrip("/")
    request = urllib.request.Request(root + "/models", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read())
    return sorted(x["id"] for x in data["data"] if isinstance(x.get("id"), str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("models", "smoke"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--plain-text", action="store_true",
                        help="Run a one-word text smoke test instead of JSON mode")
    parser.add_argument("--prompt-version", default="B-api-smoke-v1")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / "cache" / "api")
    parser.add_argument("--ledger", type=Path, default=Path(__file__).parent / "exports" / "api_usage.jsonl")
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=600)
    parser.add_argument("--max-output-tokens", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"))
    parser.add_argument("--omit-temperature", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "models":
            print("\n".join(list_models(args.base_url, args.key_env, args.timeout)))
            return 0
        client = APIClient(model=args.model, base_url=args.base_url, key_env=args.key_env,
                           cache_dir=args.cache_dir, ledger_path=args.ledger,
                           budget=Budget(args.max_requests, args.max_total_tokens),
                           timeout_s=args.timeout, reasoning_effort=args.reasoning_effort,
                           omit_temperature=args.omit_temperature)
        if args.plain_text:
            messages = [{"role": "system", "content": "Respond with one word."},
                        {"role": "user", "content": "Reply exactly OK."}]
        else:
            messages = [{"role": "system", "content": "Return only a JSON object."},
                        {"role": "user", "content": "Return {\"ok\": true}."}]
        result = client.chat(messages, prompt_version=args.prompt_version,
                             max_output_tokens=args.max_output_tokens,
                             require_json=not args.plain_text)
        if args.plain_text and result.content.strip() != "OK":
            raise RuntimeError("Plain-text smoke response did not match OK")
        if not args.plain_text and result.parsed != {"ok": True}:
            raise RuntimeError("Structured smoke response did not match expected object")
        print(json.dumps({"status": "ok", "model": args.model, "cache_hit": result.cache_hit,
                          "usage": result.usage, "cache_key": result.cache_key}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"smoke_failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
