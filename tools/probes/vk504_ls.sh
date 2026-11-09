#!/bin/sh
# What the desktop has for #504: model cache, llama.cpp builds, tools.
set -u
C=${GEIST_MODEL_CACHE:-$HOME/geist-models}
echo "cache: $C"
find "$C" -maxdepth 3 \( -iname '*deepseek*' -o -iname '*gigachat*' -o -iname '*olmoe*' \) -exec ls -la {} \; 2>/dev/null
ls -la "$C" 2>/dev/null | head -60
find "$HOME" -maxdepth 4 -iname '*.gguf' \( -iname '*deepseek*' -o -iname '*gigachat*' \) 2>/dev/null | head -40
ls -d "$HOME"/llama.cpp* 2>/dev/null
for d in "$HOME"/llama.cpp*/build-*; do echo "$d: $(cat "$d/.geist-built-from" 2>/dev/null)"; done
df -h "$C" "$HOME" 2>/dev/null
python3 -c "import gguf; print('gguf-py', gguf.__file__)" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1
uptime
