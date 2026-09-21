# Qwen3.8 Flash EXL3 OpenAI server (GGUF envelope parity)

This directory is **not** an engine patch. It is the OpenAI-compatible FastAPI
server and pure-stdlib response parser used on `:12703` so EXL3 matches the
GGUF Unc sibling (`llama-server` + Sharp template) through proxy `:12434`.

Upstream `exllamav3` can be upgraded independently. Keep this tree (and this
branch `qwen38-flash-next-decode-v2`) so the parser/server are not lost.

## Files

- `exl3_qwen38_parser.py` — `parse_completion` / `diff_parsed` (no torch)
- `exl3_qwen38_server.py` — FastAPI server wrapping exllamav3
- `test_exl3_qwen38_parity.py` — GPU-free pytest

Runtime copy on this host remains:

`/home/op/ULTIMATE/NAMEOFMODEL/exl3_qwen38_{parser,server}.py`

## Parser contract (short)

- Optional `generation_prompt` (Sharp assistant suffix) prepended like
  llama-server (`generation_prompt + generated_text`), then
  `<|im_start|>assistant\n` is consumed.
- Leading / prompt-opened `<think>` → `reasoning_content`.
- XML `<tool_call>` / `<function=` / `<parameter=` → OpenAI `tool_calls`.
- Streaming holds tool-call deltas until `</tool_call>` is closed (no ghost
  calls if the block is later truncated).
- `tool_choice: required` or a named function raises `ValueError` on a final
  parse with no matching call.

## Tests

```
python3 -m pytest qwen38_flash_gguf_parity/test_exl3_qwen38_parity.py -v
```

Sharp-template and families-JSON tests skip if those host paths are missing.
