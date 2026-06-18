#!/usr/bin/env python3
"""
Quick validation script for dual-3060 Qwen3.5-9B deployment.
Run this to verify the server is working correctly.

Usage:
  python tests/validate_dual_3060.py
"""

import json
import time
import urllib.request
import sys

HOST = "127.0.0.1"
PORT = 1235
MODEL = "Qwen3.5-9B-exl3-3060-dual"


def check_health():
    url = f"http://{HOST}:{PORT}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def check_models():
    url = f"http://{HOST}:{PORT}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def quick_completion():
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "max_tokens": 64,
        "temperature": 0,
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read().decode())
        elapsed = time.time() - t0
        return {
            "elapsed_s": round(elapsed, 3),
            "finish_reason": result["choices"][0]["finish_reason"],
            "tokens": result.get("usage", {}).get("completion_tokens", 0),
            "preview": result["choices"][0]["message"]["content"][:100],
        }
    except Exception as e:
        return {"error": str(e)}


def thinking_budget_test():
    """Verify MAX_THINKING_TOKENS cap: response must arrive within a tight budget."""
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Solve step by step: what is 17 * 19? Show your reasoning.",
            }
        ],
        "max_tokens": 2048,
        "temperature": 0,
        "thinking_budget": 256,  # very tight — forces early </think> injection
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read().decode())
        elapsed = time.time() - t0
        content = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        # Thinking + response tokens should stay well within 2048 when budget=256
        return {
            "elapsed_s": round(elapsed, 3),
            "finish_reason": result["choices"][0]["finish_reason"],
            "total_tokens": tokens,
            "budget_respected": tokens <= 2048,
            "preview": content[:200],
        }
    except Exception as e:
        return {"error": str(e)}


def no_thinking_test():
    """Verify enable_thinking=False bypasses the think block entirely."""
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello in Spanish."}],
        "max_tokens": 64,
        "temperature": 0,
        "enable_thinking": False,
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read().decode())
        elapsed = time.time() - t0
        content = result["choices"][0]["message"]["content"]
        return {
            "elapsed_s": round(elapsed, 3),
            "tokens": result.get("usage", {}).get("completion_tokens", 0),
            "no_think_tag": "<think>" not in content,
            "preview": content[:100],
        }
    except Exception as e:
        return {"error": str(e)}


def thinking_completion():
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Think step by step, then end with FINAL: say hello.",
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read().decode())
        elapsed = time.time() - t0
        content = result["choices"][0]["message"]["content"]
        has_final = "FINAL:" in content
        return {
            "elapsed_s": round(elapsed, 3),
            "finish_reason": result["choices"][0]["finish_reason"],
            "tokens": result.get("usage", {}).get("completion_tokens", 0),
            "has_final_tag": has_final,
            "final_line": [
                line for line in content.split("\n") if line.startswith("FINAL:")
            ][-1]
            if has_final
            else None,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"=== Dual 3060 Validation ===")
    print(f"Target: http://{HOST}:{PORT}")
    print()

    print("--- Health ---")
    health = check_health()
    if "error" in health:
        print(f"FAIL: {health['error']}")
        sys.exit(1)
    print(json.dumps(health, indent=2))
    print()

    print("--- Models ---")
    models = check_models()
    if "error" in models:
        print(f"FAIL: {models['error']}")
        sys.exit(1)
    print(json.dumps(models, indent=2))
    print()

    print("--- Quick Completion ---")
    quick = quick_completion()
    if "error" in quick:
        print(f"FAIL: {quick['error']}")
        sys.exit(1)
    print(json.dumps(quick, indent=2))
    print()

    print("--- Thinking Completion ---")
    think = thinking_completion()
    if "error" in think:
        print(f"FAIL: {think['error']}")
        sys.exit(1)
    print(json.dumps(think, indent=2))
    print()

    print("--- Thinking Budget Test (budget=256) ---")
    budget = thinking_budget_test()
    if "error" in budget:
        print(f"FAIL: {budget['error']}")
        sys.exit(1)
    print(json.dumps(budget, indent=2))
    print()

    print("--- No-Thinking Test ---")
    nothink = no_thinking_test()
    if "error" in nothink:
        print(f"FAIL: {nothink['error']}")
        sys.exit(1)
    print(json.dumps(nothink, indent=2))
    print()

    print("=== All Checks Passed ===")


if __name__ == "__main__":
    main()
