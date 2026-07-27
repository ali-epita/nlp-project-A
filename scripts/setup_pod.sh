#!/usr/bin/env bash
# Setup for a fresh RunPod GPU pod (tested against the PyTorch template).
# Run from the repository root after syncing the repo to the pod:
#   bash scripts/setup_pod.sh
set -euo pipefail

echo "=== System packages ==="
apt-get -qq update && apt-get -qq install -y zstd tmux rsync curl

echo "=== uv ==="
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Python environment ==="
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"

echo "=== Ollama ==="
if ! command -v ollama >/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for i in $(seq 1 30); do
        curl -s http://127.0.0.1:11434/api/tags >/dev/null && break
        sleep 1
    done
fi
curl -s http://127.0.0.1:11434/api/tags >/dev/null || { echo "Ollama failed to start"; exit 1; }

echo "=== Generation models (about 30 GB of pulls) ==="
for model in \
    llama3.1:8b-instruct-q4_K_M \
    mistral:7b-instruct \
    qwen2.5:7b-instruct \
    phi4:14b \
    qwen2.5:14b-instruct-q4_K_M; do
    ollama pull "$model"
done
ollama list

echo "=== Done. Suggested order ==="
echo "  uv run finrag download && uv run finrag extract"
echo "  uv run python experiments/run_grid.py --sweep all --device cuda"
echo "  uv run python experiments/analyze_retrieval.py"
echo "  uv run python experiments/run_generation.py --all -k 10"
