#!/usr/bin/env python3
r"""Pure response parser for the EXL3 Qwen3.8-Flash-Next backend (:12703).

Mirrors the GGUF sibling's (llama-server :12702) chat response envelope for
the Qwen3-Coder output parser that the Sharp template selects.  Stdlib only
(no model-runtime or web-framework imports, no import-time side effects) so
offline tests and the exl3 server share one code path.

Contract implemented by ``parse_completion`` (see the parity contract in the
``exl3_qwen38_server.py`` docstring):

* a leading ``<think>`` block becomes ``reasoning_content``.  It ends at
  ``</think>`` (consumed) or at ``<tool_call>`` (not consumed), whichever
  comes first; one whitespace run after ``<think>`` is consumed
  (llama-server ``p.space()``); unclosed at EOF still counts.  Without a
  leading ``<think>`` reasoning is ``""``;
* content runs from the end of the reasoning block up to the first
  ``<tool_call>`` (lstripped); with no tool region it is the remaining text;
* XML ``<tool_call>`` blocks
  (``<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n``
  ``</function>\n</tool_call>``) become OpenAI ``tool_calls`` with
  ``finish_reason`` ``"tool_calls"``.  ``NAME`` must match a request tool,
  ``ARG`` must be declared by that tool's ``parameters.properties`` when the
  schema declares properties (any name otherwise), and the schema ``required``
  list must be present.  Values stay raw strings and ``arguments`` is
  ``json.dumps`` of the ``{ARG: VALUE}`` object.  Sequential blocks get
  indexes 0, 1, 2, ...
* a tool region that cannot be fully validated (unknown function, malformed
  markup, missing required parameter, undeclared parameter) yields content
  truncated at the first ``<tool_call>``, ``tool_calls = []`` and
  ``finish_reason`` ``"stop"`` -- never an exception, except for a forced
  ``tool_choice`` on a final parse (below).

Generation prompt (llama-server parity)
=======================================

llama-server parses ``generation_prompt + generated_text``.  The Sharp
template already emits ``<|im_start|>assistant\\n`` plus an opened or
pre-closed ``<think>`` block; the model continues inside that.  Pass that
suffix as ``generation_prompt`` so thinking output without a leading
``<think>`` still becomes ``reasoning_content``.  The prefix
``<|im_start|>assistant\\n`` is consumed and is never content.

Streaming (``partial=True``)
============================

``parse_completion`` is re-called on the growing accumulated text.  In
partial mode the longest trailing proper prefix of a recognized tag
(``<think>``, ``</think>``, ``<tool_call>``, ``</tool_call>``,
``<function=``, ``</function>``, ``<parameter=``, ``</parameter>``) is held
back so a half-written tag never leaks into reasoning/content.  Incomplete
``<tool_call>`` blocks are held until ``</tool_call>`` is closed — no
tool-call deltas until then, so a later failed/truncated region cannot leave
a ghost call.  Reasoning and content still stream.  A trailing block already
known invalid still follows the failure semantics above.  Partial mode never
raises, including for a forced ``tool_choice``.

Tool call ids
=============

Ids are deterministic (``"call_" + sha1("<index>\\0<name>")``) so successive
re-parses of the same response produce identical ids and ``diff_parsed``
never regenerates them.  A call only appears once its function name is
complete, so (index, name) is stable for the lifetime of a response.

``diff_parsed``
===============

Deltas are ordered: reasoning suffix, content suffix, then tool-call deltas.
The first appearance of index ``i`` carries ``id``/``type``/``name`` plus the
arguments accumulated so far; later updates for the same index carry only
``arguments``.  Suffix semantics follow llama.cpp ``string_diff``: empty
previous -> full current; current starts with previous -> suffix; previous
starts with current (shrunk) -> no delta.  Unlike llama.cpp this module never
throws on a shrinking or non-prefix segment: it emits no delta for that
segment and the caller keeps the previous value.

Other deliberate choices
========================

* ASCII whitespace (``std::isspace`` C locale) is used for the ``<think>``
  lead-in run and for whitespace between tool blocks; content lstrip uses
  Python ``str.lstrip()``.
* Text after the last complete tool block is ignored: it never becomes
  content and never invalidates the region.
* Duplicate parameter names are kept as separate JSON entries.
* ``tool_choice == "required"`` or a dict naming a function (llama-server
  maps that to "required") raises ``ValueError`` on a final parse when no
  valid call was produced; partial parses never raise.
* With ``tools`` empty/None a tool region cannot validate, so it follows the
  failure semantics (the server does not pass tools for ``tool_choice:
  "none"``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["parse_completion", "diff_parsed"]

# ASCII whitespace, matching std::isspace in the C locale.
_WS = " \t\r\n\f\v"

# Recognized markup; partial trailing proper prefixes are held back while
# streaming so half-written tags never leak as content.
_TAGS: tuple[str, ...] = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)

_GEN_PREFIX = "<|im_start|>assistant\n"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_REASONING_END: tuple[str, ...] = (_THINK_CLOSE, "<tool_call>")
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE_SEQ = "\n</tool_call>"
_FUNC_OPEN = "<function="
_FUNC_CLOSE = "</function>"
_ARG_OPEN = "<parameter="
_ARG_CLOSE = "\n</parameter>"


# ---------------------------------------------------------------- helpers

def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _tool_specs(tools: Any) -> dict[str, tuple[frozenset[str] | None, tuple[str, ...]]]:
    """Normalize request tools to {name: (declared_properties | None, required)}."""
    specs: dict[str, tuple[frozenset[str] | None, tuple[str, ...]]] = {}
    if not isinstance(tools, (list, tuple)):
        return specs
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            fn = tool  # tolerate a flat {"name": ..., "parameters": ...}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        props: frozenset[str] | None = None
        required: tuple[str, ...] = ()
        params = fn.get("parameters")
        if isinstance(params, dict):
            declared = params.get("properties")
            if isinstance(declared, dict) and declared:
                props = frozenset(str(key) for key in declared)
            req = params.get("required")
            if isinstance(req, (list, tuple)):
                required = tuple(item for item in req if isinstance(item, str))
        specs[name] = (props, required)
    return specs


def _required_tool_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name:
                return name
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return name
    return None


def _forced_choice(tool_choice: Any) -> bool:
    """True when a missing/failed tool call must raise on the final parse."""
    if tool_choice == "required":
        return True
    return _required_tool_name(tool_choice) is not None


def _hold_partial_tail(text: str) -> tuple[str, str]:
    """Split off a trailing half-written tag so it never leaks while streaming."""
    limit = min(len(text), max(len(tag) for tag in _TAGS) - 1)
    for size in range(limit, 0, -1):
        suffix = text[-size:]
        for tag in _TAGS:
            if len(suffix) < len(tag) and tag.startswith(suffix):
                return text[:-size], suffix
    return text, ""


def _is_partial_tag(text: str, tag: str) -> bool:
    return len(text) < len(tag) and tag.startswith(text)


def _find_reasoning_end(text: str, start: int) -> tuple[int | None, str | None]:
    best: int | None = None
    needle: str | None = None
    for tag in _REASONING_END:
        found = text.find(tag, start)
        if found != -1 and (best is None or found < best):
            best, needle = found, tag
    return best, needle


def _trailing_prefix_len(text: str, term: str) -> int:
    """Longest k < len(term) with text ending in term[:k] (0 when none)."""
    for size in range(len(term) - 1, 0, -1):
        if text.endswith(term[:size]):
            return size
    return 0


def _call_id(index: int, name: str) -> str:
    """Deterministic id: stable across re-parses of the same response."""
    digest = hashlib.sha1(f"{index}\x00{name}".encode("utf-8")).hexdigest()
    return "call_" + digest[:32]


def _args_json(entries: list[tuple[str, str, bool]], closed: bool) -> str:
    """Build the arguments string, growing as a strict prefix while streaming."""
    parts = []
    for name, value, value_closed in entries:
        encoded = json.dumps(value, ensure_ascii=True)
        if not value_closed:
            encoded = encoded[:-1]  # drop the closing quote of an open value
        parts.append(json.dumps(name, ensure_ascii=True) + ": " + encoded)
    return "{" + ", ".join(parts) + ("}" if closed else "")


def _make_call(
    index: int,
    name: str,
    entries: list[tuple[str, str, bool]],
    closed: bool,
    specs: dict[str, tuple[frozenset[str] | None, tuple[str, ...]]],
) -> dict[str, Any] | None:
    """Build an OpenAI tool call, or None when the block fails validation."""
    spec = specs.get(name)
    if spec is None:
        return None
    props, required = spec
    arg_names = [entry[0] for entry in entries]
    if props is not None and any(arg not in props for arg in arg_names):
        return None
    if closed and any(req not in arg_names for req in required):
        return None
    return {
        "id": _call_id(index, name),
        "type": "function",
        "function": {"name": name, "arguments": _args_json(entries, closed)},
    }


def _parse_tool_region(
    text: str,
    start: int,
    specs: dict[str, tuple[frozenset[str] | None, tuple[str, ...]]],
    partial: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Parse sequential <tool_call> blocks starting at ``start``.

    Returns ``(calls, failed)``.  ``failed`` is True when the region is known
    invalid; the caller then discards ``calls`` (all-or-nothing semantics).
    Incomplete trailing blocks are held when ``partial`` is True (no call
    until ``</tool_call>``) and fail the region on the final parse.
    """
    calls: list[dict[str, Any]] = []
    i = start
    n = len(text)
    while i < n and text.startswith(_TOOL_OPEN, i):
        j = i + len(_TOOL_OPEN)
        if j >= n or text[j] != "\n":
            if j >= n and partial:
                return calls, False
            return calls, True
        j += 1
        if not text.startswith(_FUNC_OPEN, j):
            if partial and _is_partial_tag(text[j:], _FUNC_OPEN):
                return calls, False
            return calls, True
        k = text.find(">", j + len(_FUNC_OPEN))
        if k == -1:
            return calls, not partial
        name = text[j + len(_FUNC_OPEN):k]
        if k + 1 >= n:
            return calls, not partial
        if text[k + 1] != "\n":
            return calls, True
        p = k + 2
        entries: list[tuple[str, str, bool]] = []
        closed = False
        incomplete = False
        while True:
            if text.startswith(_ARG_OPEN, p):
                q = text.find(">", p + len(_ARG_OPEN))
                if q == -1:
                    incomplete = True
                    break
                arg = text[p + len(_ARG_OPEN):q]
                if q + 1 >= n:
                    incomplete = True
                    break
                if text[q + 1] != "\n":
                    return calls, True
                vstart = q + 2
                t = text.find(_ARG_CLOSE, vstart)
                if t == -1:
                    seg = text[vstart:]
                    hold = _trailing_prefix_len(seg, _ARG_CLOSE)
                    entries.append((arg, seg[:len(seg) - hold] if hold else seg, False))
                    incomplete = True
                    break
                entries.append((arg, text[vstart:t], True))
                p = t + len(_ARG_CLOSE)
                if p >= n:
                    incomplete = True
                    break
                if text[p] != "\n":
                    return calls, True
                p += 1
                continue
            if text.startswith(_FUNC_CLOSE, p):
                closed = True
                p += len(_FUNC_CLOSE)
                break
            if partial and (_is_partial_tag(text[p:], _ARG_OPEN)
                            or _is_partial_tag(text[p:], _FUNC_CLOSE)):
                incomplete = True
                break
            return calls, True
        if incomplete or not closed:
            return calls, not partial
        if text.startswith(_TOOL_CLOSE_SEQ, p):
            p += len(_TOOL_CLOSE_SEQ)
        elif partial and _is_partial_tag(text[p:], _TOOL_CLOSE_SEQ):
            return calls, False
        else:
            return calls, True
        call = _make_call(len(calls), name, entries, True, specs)
        if call is None:
            return calls, True
        calls.append(call)
        while p < n and text[p] in _WS:
            p += 1
        i = p
    return calls, False


# ---------------------------------------------------------------- public API

def parse_completion(
    text: str,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    partial: bool = False,
    generation_prompt: str = "",
) -> dict:
    """Parse accumulated generation text into the llama-server response envelope.

    Returns exactly ``{"reasoning_content", "content", "tool_calls",
    "finish_reason"}``; see the module docstring for the full contract.
    """
    text = _as_text(generation_prompt) + _as_text(text)
    if text.startswith(_GEN_PREFIX):
        text = text[len(_GEN_PREFIX):]
    specs = _tool_specs(tools)
    forced = _forced_choice(tool_choice)
    required_name = _required_tool_name(tool_choice)
    if partial:
        text, _held = _hold_partial_tail(text)

    reasoning = ""
    pos = 0
    if text.startswith(_THINK_OPEN):
        pos = len(_THINK_OPEN)
        while pos < len(text) and text[pos] in _WS:
            pos += 1
        end, terminator = _find_reasoning_end(text, pos)
        if end is None:
            reasoning = text[pos:]  # unclosed at EOF still counts
            pos = len(text)
        else:
            reasoning = text[pos:end]
            pos = end + len(_THINK_CLOSE) if terminator == _THINK_CLOSE else end

    content = ""
    tool_pos: int | None = None
    idx = text.find(_TOOL_OPEN, pos)
    if idx == -1:
        content = text[pos:].lstrip()
    else:
        content = text[pos:idx].lstrip()
        tool_pos = idx

    tool_calls: list[dict[str, Any]] = []
    if tool_pos is not None:
        calls, failed = _parse_tool_region(text, tool_pos, specs, partial)
        if not failed:
            tool_calls = calls
    if required_name:
        tool_calls = [
            call for call in tool_calls
            if ((call.get("function") or {}).get("name") == required_name)
        ]

    if forced and not tool_calls and not partial:
        raise ValueError(
            "tool_choice requires a tool call but the model output did not "
            "contain a valid <tool_call> block"
        )

    return {
        "reasoning_content": reasoning,
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }


def _string_diff(last: str, current: str) -> str:
    """llama.cpp string_diff, minus the exceptions (no delta instead)."""
    if not last:
        return current
    if current.startswith(last):
        return current[len(last):]
    if last.startswith(current):
        return ""  # shrunk (partial stop word); keep the previous value
    return ""  # non-prefix jump; never raise in production


def _call_arguments(call: Any) -> str:
    fn = (call or {}).get("function") or {}
    args = fn.get("arguments")
    return args if isinstance(args, str) else ""


def diff_parsed(prev: dict, cur: dict) -> list[dict]:
    """Prefix-diff two parse results into ordered OpenAI streaming deltas."""
    diffs: list[dict] = []

    delta = _string_diff(
        _as_text(prev.get("reasoning_content")), _as_text(cur.get("reasoning_content"))
    )
    if delta:
        diffs.append({"reasoning_content": delta})

    delta = _string_diff(_as_text(prev.get("content")), _as_text(cur.get("content")))
    if delta:
        diffs.append({"content": delta})

    prv_calls = prev.get("tool_calls") or []
    cur_calls = cur.get("tool_calls") or []

    for index in range(min(len(prv_calls), len(cur_calls))):
        delta = _string_diff(_call_arguments(prv_calls[index]), _call_arguments(cur_calls[index]))
        if delta:
            diffs.append({"tool_calls": [{"index": index, "function": {"arguments": delta}}]})

    for index in range(len(prv_calls), len(cur_calls)):
        call = cur_calls[index] or {}
        fn = call.get("function") or {}
        entry: dict[str, Any] = {"index": index}
        if call.get("id"):
            entry["id"] = call["id"]
            entry["type"] = "function"
        entry["function"] = {"name": fn.get("name", "")}
        args = fn.get("arguments")
        if isinstance(args, str) and args:
            entry["function"]["arguments"] = args
        diffs.append({"tool_calls": [entry]})

    return diffs
