import os

# "cpu" or "cuda"
DEVICE = os.environ.get("DEVICE", "cpu")

# how many regions processed concurrently by VLM agents. keep low for
# a single local GPU/CPU -- true speedup is limited since compute
# still serializes on one device; mainly hides Python/IO overhead.
# raise this only if you're calling a remote/API-based VLM.
VLM_MAX_WORKERS = int(os.environ.get("VLM_MAX_WORKERS", "2"))

# retry attempts (not counting the first try) per VLM call
VLM_RETRIES = int(os.environ.get("VLM_RETRIES", "2"))

# seconds to wait for a single generate() call before treating it as failed
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "180"))
