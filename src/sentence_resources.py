"""Starter CPU model setup: explicit download, local-only normal loading.

Run ``python -m src.sentence_resources --download`` before the workshop.
Run without --download to verify the cache without contacting the Hub.
No credentials, remote inference, or substitute embeddings are used.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import time

SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Freeze the validated public model snapshot for reproducible workshop preloads.
SENTENCE_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIMENSION = 384
SENTENCE_TOKEN_LIMIT = 256
MODEL_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "sentence_transformers"


class SentenceModelUnavailable(RuntimeError):
    """Readable setup failure for notebooks and the Streamlit lab."""


@lru_cache(maxsize=1)
def _load_cached_tokenizer(cache_directory: str, revision: str):
    # Task 11 length analysis needs the same tokenizer, not encoder weights.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot = snapshot_download(SENTENCE_MODEL_ID, cache_dir=cache_directory,
                                 revision=revision, local_files_only=True, token=False)
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def load_sentence_tokenizer(cache_directory=None):
    """Starter local-only MiniLM tokenizer for length checks, without inference."""
    cache = Path(cache_directory or MODEL_CACHE).resolve()
    try:
        return _load_cached_tokenizer(str(cache), SENTENCE_MODEL_REVISION)
    except Exception as error:
        raise SentenceModelUnavailable(
            f"Cannot load the shared MiniLM tokenizer from {cache}. Preload while online with "
            "python -m src.sentence_resources --download, then run offline. "
            "Document extraction, cleaning and chunking remain usable without it. "
            f"Setup detail: {type(error).__name__}: {error}"
        ) from error


@lru_cache(maxsize=1)
def _load_cached_model(cache_directory: str, revision: str):
    # Imports are lazy so a missing scientific dependency cannot break other labs.
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    import torch

    snapshot = snapshot_download(SENTENCE_MODEL_ID, cache_dir=cache_directory,
                                 revision=revision, local_files_only=True, token=False)
    # A local path avoids remote adapter/config probes during ordinary loading.
    model = SentenceTransformer(snapshot, device="cpu", local_files_only=True,
                                trust_remote_code=False)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model.eval()
    if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSION:
        raise ValueError("Expected the shared MiniLM model's 384 dimensions.")
    return model


def load_sentence_embedding_model(cache_directory=None, *, allow_download=False):
    """Load exact shared encoder on CPU. Downloads require explicit opt-in.

    Local cache path can be supplied for preflight/offline transfer. Successful
    loads are reused within a process; failed loads are not cached. The public
    model uses no Hugging Face token. First download requires internet access.
    """
    cache = Path(cache_directory or MODEL_CACHE).resolve()
    try:
        if allow_download:
            from huggingface_hub import snapshot_download
            cache.mkdir(parents=True, exist_ok=True)
            snapshot_download(SENTENCE_MODEL_ID, cache_dir=str(cache), token=False,
                              revision=SENTENCE_MODEL_REVISION,
                              allow_patterns=["*.json", "*.txt", "*.safetensors"])
            _load_cached_model.cache_clear()
        return _load_cached_model(str(cache), SENTENCE_MODEL_REVISION)
    except Exception as error:
        raise SentenceModelUnavailable(
            f"Cannot load {SENTENCE_MODEL_ID} from {cache}. "
            "From the repository root, run python -m src.sentence_resources --download "
            "while online, then python -m src.sentence_resources. "
            "For offline use, copy the complete cache directory from a prepared machine. "
            "Check installed requirements, disk space and cache permissions. "
            f"Setup detail: {type(error).__name__}: {error}"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicit public-model download")
    parser.add_argument("--cache-dir", type=Path, default=MODEL_CACHE)
    args = parser.parse_args()
    if args.download:
        print(f"Downloading public {SENTENCE_MODEL_ID}; no account/token required.")
    started = time.perf_counter()
    try:
        model = load_sentence_embedding_model(args.cache_dir, allow_download=args.download)
        loaded = time.perf_counter()
        from src.embeddings import encode_sentences
        vectors = encode_sentences(model, ["Students learn natural language processing."])
        elapsed = time.perf_counter() - loaded
    except SentenceModelUnavailable as error:
        parser.exit(1, str(error) + "\n")
    print(f"CPU preflight OK: shape={vectors.shape}, finite inference, no paid credentials.")
    print(f"Load/setup: {loaded-started:.3f}s; one-sentence encoding: {elapsed:.3f}s")
    print(f"Cache: {args.cache_dir.resolve()}")
    print(f"Cache file bytes: {sum(p.stat().st_size for p in args.cache_dir.rglob('*') if p.is_file())}")


if __name__ == "__main__":
    main()
