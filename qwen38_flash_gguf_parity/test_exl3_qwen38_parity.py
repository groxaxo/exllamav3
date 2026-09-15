import json
import sys
from pathlib import Path

import pytest
from jinja2 import BaseLoader, Environment


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from exl3_qwen38_parser import diff_parsed, parse_completion  # noqa: E402


SHARP_TEMPLATE = Path("/home/op/models/qwen-sharp-chat-templates/chat_template.jinja")
FAMILIES_FILE = Path("/home/op/ULTIMATE/NAMEOFMODEL/qwen36_model_families.json")
EXL3_FAMILY = "qwen38-flash-next-unc-exl3-4bpw"
GGUF_FAMILY = "qwen38-flash-next-unc-q4km-agentic-v2"

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

VALID_TOOL_XML = (
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=city>\n"
    "Paris\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

THINK_GEN = "<|im_start|>assistant\n<think>\n"
INSTRUCT_GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _render_sharp(messages, *, enable_thinking, preserve_thinking):
    env = Environment(loader=BaseLoader(), autoescape=False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    template = env.from_string(SHARP_TEMPLATE.read_text())
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
    )


def test_parse_think_only() -> None:
    parsed = parse_completion("<think>a</think>", partial=False)
    assert parsed["reasoning_content"] == "a"
    assert parsed["content"] == ""
    assert parsed["tool_calls"] == []
    assert parsed["finish_reason"] == "stop"


def test_parse_think_and_content() -> None:
    parsed = parse_completion("<think>a</think>\n\nHello", partial=False)
    assert parsed["reasoning_content"] == "a"
    assert parsed["content"] == "Hello"
    assert parsed["tool_calls"] == []
    assert parsed["finish_reason"] == "stop"


def test_parse_think_content_and_valid_tool_call() -> None:
    text = "<think>a</think>\n\nHello" + VALID_TOOL_XML
    parsed = parse_completion(text, tools=WEATHER_TOOLS, partial=False)
    assert parsed["reasoning_content"] == "a"
    assert parsed["content"] == "Hello"
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert parsed["finish_reason"] == "tool_calls"


def test_parse_unclosed_think_at_eof() -> None:
    parsed = parse_completion("<think>a", partial=False)
    assert parsed["reasoning_content"] == "a"
    assert parsed["content"] == ""
    assert parsed["tool_calls"] == []
    assert parsed["finish_reason"] == "stop"


def test_parse_think_ending_at_tool_call() -> None:
    text = "<think>a" + VALID_TOOL_XML
    parsed = parse_completion(text, tools=WEATHER_TOOLS, partial=False)
    assert parsed["reasoning_content"] == "a"
    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Paris"
    }
    assert parsed["finish_reason"] == "tool_calls"


def test_parse_non_matching_tool_region() -> None:
    unknown = (
        "<think>a</think>\n\nHello"
        "<tool_call>\n"
        "<function=not_a_tool>\n"
        "<parameter=city>\n"
        "Paris\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    malformed = (
        "<think>a</think>\n\nHello"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city\n"
        "Paris\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    for text in (unknown, malformed):
        parsed = parse_completion(text, tools=WEATHER_TOOLS, partial=False)
        assert parsed["content"] == "Hello"
        assert parsed["tool_calls"] == []
        assert parsed["finish_reason"] == "stop"


def test_parse_tool_choice_required_raises() -> None:
    with pytest.raises(ValueError):
        parse_completion("Hello", tools=WEATHER_TOOLS, tool_choice="required", partial=False)


def test_parse_plain_content_passthrough() -> None:
    parsed = parse_completion("Hello", partial=False)
    assert parsed["reasoning_content"] == ""
    assert parsed["content"] == "Hello"
    assert parsed["tool_calls"] == []
    assert parsed["finish_reason"] == "stop"


def test_partial_split_tag_safety() -> None:
    prefixes = (
        "He<thi",
        "<think>th",
        "<tool_cal",
        "<tool_call>\n<function=foo>\n<parameter=ar",
    )
    for prefix in prefixes:
        parsed = parse_completion(prefix, tools=WEATHER_TOOLS, partial=True)
        blob = parsed["reasoning_content"] + parsed["content"]
        assert "<thi" not in blob
        assert "<tool_cal" not in blob
        assert "<parameter=" not in blob


def test_diff_parsed_streaming_order_and_stable_ids() -> None:
    prefixes = [
        "<think>rea",
        "<think>reason",
        "<think>reason</think>",
        "<think>reason</think>\n\nHel",
        "<think>reason</think>\n\nHello",
        "<think>reason</think>\n\nHello<tool_call>\n<function=get_weather>\n",
        "<think>reason</think>\n\nHello<tool_call>\n<function=get_weather>\n<parameter=city>\nP",
        "<think>reason</think>\n\nHello<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
        "<think>reason</think>\n\nHello<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
        "<think>reason</think>\n\nHello<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    ]
    prev = {"reasoning_content": "", "content": "", "tool_calls": []}
    deltas: list[dict] = []
    last = prev
    for text in prefixes:
        cur = parse_completion(text, tools=WEATHER_TOOLS, partial=True)
        if "</tool_call>" in text:
            assert len(cur["tool_calls"]) == 1
        else:
            assert cur["tool_calls"] == []
        deltas.extend(diff_parsed(prev, cur))
        prev = cur
        last = cur

    kinds: list[str] = []
    for delta in deltas:
        if "reasoning_content" in delta:
            kinds.append("reasoning")
        elif "content" in delta:
            kinds.append("content")
        elif "tool_calls" in delta:
            kinds.append("tool_calls")
        else:
            raise AssertionError(f"unexpected delta {delta!r}")
    rank = {"reasoning": 0, "content": 1, "tool_calls": 2}
    assert kinds == sorted(kinds, key=rank.__getitem__)

    tool_deltas = [delta["tool_calls"] for delta in deltas if "tool_calls" in delta]
    assert len(tool_deltas) == 1
    assert len(tool_deltas[0]) == 1
    entry = tool_deltas[0][0]
    final_calls = last["tool_calls"]
    assert len(final_calls) == 1
    assert entry["index"] == 0
    assert entry.get("id")
    assert entry["id"] == final_calls[0]["id"]
    fn = entry.get("function") or {}
    assert fn.get("name") == "get_weather"
    assert fn.get("arguments") == final_calls[0]["function"]["arguments"]
    assert json.loads(final_calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_parse_thinking_generation_prompt() -> None:
    parsed = parse_completion(
        "plan</think>\n\nHi", generation_prompt=THINK_GEN, partial=False
    )
    assert parsed["reasoning_content"] == "plan"
    assert parsed["content"] == "Hi"
    assert parsed["tool_calls"] == []
    assert parsed["finish_reason"] == "stop"


def test_parse_instruct_generation_prompt() -> None:
    parsed = parse_completion("Hi", generation_prompt=INSTRUCT_GEN, partial=False)
    assert parsed["reasoning_content"] == ""
    assert parsed["content"] == "Hi"
    assert parsed["tool_calls"] == []
    assert parsed["finish_reason"] == "stop"


def test_partial_valid_tool_xml_held_until_close() -> None:
    for i in range(1, len(VALID_TOOL_XML) + 1):
        prefix = VALID_TOOL_XML[:i]
        parsed = parse_completion(prefix, tools=WEATHER_TOOLS, partial=True)
        if prefix.endswith("</tool_call>"):
            assert len(parsed["tool_calls"]) == 1
            call = parsed["tool_calls"][0]
            assert call["function"]["name"] == "get_weather"
            assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        else:
            assert parsed["tool_calls"] == []


def test_stream_then_truncated_eof_no_ghost_calls() -> None:
    text = "<think>a</think>\n\nHello" + VALID_TOOL_XML[: -len("</tool_call>")]
    assert "</tool_call>" not in text
    prev = {"reasoning_content": "", "content": "", "tool_calls": []}
    growing = ""
    for ch in text:
        growing += ch
        cur = parse_completion(growing, tools=WEATHER_TOOLS, partial=True)
        assert cur["tool_calls"] == []
        for delta in diff_parsed(prev, cur):
            assert "tool_calls" not in delta
        prev = cur
    parsed = parse_completion(text, tools=WEATHER_TOOLS, partial=False)
    assert parsed["tool_calls"] == []
    assert parsed["finish_reason"] == "stop"
    assert parsed["content"] == "Hello"
    assert parsed["reasoning_content"] == "a"


def test_parse_tool_choice_named_function_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        parse_completion(
            VALID_TOOL_XML,
            tools=WEATHER_TOOLS,
            tool_choice={"type": "function", "function": {"name": "bar"}},
            partial=False,
        )


@pytest.mark.skipif(not SHARP_TEMPLATE.is_file(), reason="Sharp template not on this host")
def test_sharp_template_enable_thinking_markers() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    off = _render_sharp(messages, enable_thinking=False, preserve_thinking=False)
    on = _render_sharp(messages, enable_thinking=True, preserve_thinking=False)
    assert off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert on.endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.skipif(not SHARP_TEMPLATE.is_file(), reason="Sharp template not on this host")
def test_sharp_template_preserve_thinking() -> None:
    marker = "UNIQUE_PRIOR_REASONING"
    messages = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "Visible answer",
            "reasoning_content": marker,
        },
        {"role": "user", "content": "Again"},
    ]
    kept = _render_sharp(messages, enable_thinking=False, preserve_thinking=True)
    dropped = _render_sharp(messages, enable_thinking=False, preserve_thinking=False)
    assert f"<think>\n{marker}\n</think>" in kept
    assert marker not in dropped


@pytest.mark.skipif(not FAMILIES_FILE.is_file(), reason="model families JSON not on this host")
def test_config_preset_overrides_parity() -> None:
    registry = json.loads(FAMILIES_FILE.read_text())
    exl3 = registry[EXL3_FAMILY]["preset_overrides"]
    gguf = registry[GGUF_FAMILY]["preset_overrides"]
    assert exl3["instruct"]["forced"] == gguf["instruct"]["forced"]
    assert exl3["thinking"]["forced"] == gguf["thinking"]["forced"]
