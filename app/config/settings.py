import os

# "cpu" or "cuda" -- override with: set DEVICE=cuda (Windows) / export DEVICE=cuda
DEVICE = os.environ.get("DEVICE", "cpu")
