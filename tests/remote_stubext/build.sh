#!/bin/sh
# Builds a no-op exllamav3_ext so the surgery test can import exllamav3 without CUDA
cd "$(dirname "$0")" && gcc -shared -fPIC $(python3-config --includes) stub.c -o "exllamav3_ext$(python3-config --extension-suffix)"
