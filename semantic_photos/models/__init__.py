import os

from huggingface_hub.constants import default_cache_path

HUGGINGFACE_CACHE = os.getenv("MODEL_CACHE_DIR", default_cache_path)

if "MODEL_CACHE_DIR" in os.environ:
    if not os.path.isdir(HUGGINGFACE_CACHE):
        os.mkdir(HUGGINGFACE_CACHE)

os.environ["HF_HOME"] = HUGGINGFACE_CACHE
os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_CACHE
