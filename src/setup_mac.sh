#!/bin/bash
# One-time setup on the Mac (Apple-silicon GPU = "MPS"). Creates its own environment; does NOT touch bioenv.
set -e
cd "$(dirname "$0")"
echo "== 1/3  Installing 'uv' (a fast Python installer) if needed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "== 2/3  Creating environment ./oral_env with Python 3.11"
uv venv --python 3.11 oral_env
echo "== 3/3  Installing PyTorch + libraries (about 1 GB, a few minutes)"
uv pip install --python oral_env/bin/python torch torchvision "timm==1.0.31" "peft==0.21.2" grad-cam pandas scikit-learn pillow
oral_env/bin/python - <<'PY'
import torch, timm, peft
print("torch", torch.__version__, "| timm", timm.__version__, "| peft", peft.__version__)
print("Apple GPU (MPS) available:", torch.backends.mps.is_available())
PY
echo "Setup finished."
