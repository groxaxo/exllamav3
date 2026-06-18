#!/bin/bash

# Pipes the URL list directly into aria2c. No temp files needed.
printf "%s\n" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/.gitattributes" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/README.md" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/chat_template.jinja" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/config.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/generation_config.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/merges.txt" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/model-00001-of-00003.safetensors" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/model-00002-of-00003.safetensors" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/model-00003-of-00003.safetensors" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/model.safetensors.index.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/preprocessor_config.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/quantization_config.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/tokenizer.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/tokenizer_config.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/video_preprocessor_config.json" \
"https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3/resolve/6.00bpw/vocab.json" \
| aria2c -i - -j 5 -x 16 -s 16 -c
