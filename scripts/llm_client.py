"""Thin client for the local vLLM OpenAI-compatible server. Logs one telemetry
event per call in the same shape ag23-llm already uses, so this dataset's
telemetry.jsonl is consistent with the rest of the portfolio."""
import os, time, json, httpx

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
_model_cache = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TELEMETRY_PATH = os.path.join(DATA_DIR, "telemetry.jsonl")

def discover_model():
    global _model_cache
    if _model_cache:
        return _model_cache
    r = httpx.get(f"{BASE_URL}/models", timeout=10)
    r.raise_for_status()
    _model_cache = r.json()["data"][0]["id"]
    return _model_cache

def call_llm(prompt, *, role, goal, project, iteration, max_tokens=2000, temperature=0.3, timeout=120):
    model = discover_model()
    t0 = time.time()
    ok, error, text, total_tokens = True, None, "", 0
    try:
        r = httpx.post(f"{BASE_URL}/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data["choices"][0]["message"].get("content") or "")
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens", 0)
    except Exception as e:
        ok = False
        error = repr(e)
    latency_ms = (time.time() - t0) * 1000
    telemetry_row = {
        "ts": round(time.time(), 3), "provider": "local-vllm", "model": model,
        "task": "coding" if role == "worker" else "review",
        "cluster": role, "latency_ms": round(latency_ms, 1), "ok": ok, "error": error,
        "total_tokens": total_tokens,
        "ctx": {"project": project, "goal": (goal or "")[:300], "iteration": iteration, "role": role},
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TELEMETRY_PATH, "a") as f:
        f.write(json.dumps(telemetry_row) + "\n")
    return {"text": text, "ok": ok, "error": error, "telemetry_row": telemetry_row}

def _clean_assistant_msg(message):
    m = {"role": message.get("role", "assistant"), "content": message.get("content") or ""}
    tc = message.get("tool_calls")
    if tc:
        m["tool_calls"] = tc
    return m

def call_llm_tools(messages, *, role, goal, project, iteration, tools=None, tool_choice="auto",
                    max_tokens=1000, temperature=0.3, timeout=120):
    """Like call_llm, but supports a full multi-turn `messages` list and an OpenAI-style
    `tools` schema. Returns the raw assistant message dict (content + tool_calls) so the
    caller can append it back into the conversation and, if there are tool_calls, execute
    them and continue the loop."""
    model = discover_model()
    t0 = time.time()
    ok, error, message, total_tokens = True, None, None, 0
    try:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        r = httpx.post(f"{BASE_URL}/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens", 0)
    except Exception as e:
        ok = False
        error = repr(e)
        message = {"role": "assistant", "content": ""}
    latency_ms = (time.time() - t0) * 1000
    telemetry_row = {
        "ts": round(time.time(), 3), "provider": "local-vllm", "model": model,
        "task": "coding" if role == "worker" else "review",
        "cluster": role, "latency_ms": round(latency_ms, 1), "ok": ok, "error": error,
        "total_tokens": total_tokens,
        "ctx": {"project": project, "goal": (goal or "")[:300], "iteration": iteration, "role": role},
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TELEMETRY_PATH, "a") as f:
        f.write(json.dumps(telemetry_row) + "\n")
    return {"message": _clean_assistant_msg(message) if message else None, "ok": ok, "error": error,
            "telemetry_row": telemetry_row}
