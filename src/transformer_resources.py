"""Starter resources: pinned public CPU pipelines, explicit preload, local use.

Run ``python -m src.transformer_resources --download`` before the workshop.
Without --download, verification is offline. No student model-loading TODOs.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

MODEL_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "transformer_tasks"
MODELS = {
    "sentiment": {"id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                  "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
                  "task": "sentiment-analysis", "license": "Apache-2.0"},
    "ner": {"id": "dslim/distilbert-NER",
            "revision": "dfa2838a127384aabb82ed7719e16dab84c42a2a",
            "task": "token-classification", "license": "Apache-2.0"},
    "qa": {"id": "distilbert/distilbert-base-uncased-distilled-squad",
           "revision": "dfb8fa0e03905f0b6ea92133e56b6fef138015f2",
           "task": "question-answering", "license": "Apache-2.0"},
}


class TransformerModelUnavailable(RuntimeError):
    """Actionable model setup error; manual attention remains usable."""


def model_directory(task, cache_directory=None):
    if task not in MODELS:
        raise ValueError("Choose sentiment, ner or qa.")
    return Path(cache_directory or MODEL_CACHE).resolve() / task / MODELS[task]["revision"]


@lru_cache(maxsize=3)
def _load_local_pipeline(task, directory):
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              AutoModelForTokenClassification, AutoModelForQuestionAnswering, pipeline)
    import torch
    classes = {"sentiment": AutoModelForSequenceClassification, "ner": AutoModelForTokenClassification,
               "qa": AutoModelForQuestionAnswering}
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = classes[task].from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True)
    model.eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    options = {"aggregation_strategy": "first"} if task == "ner" else {}
    return pipeline(MODELS[task]["task"], model=model, tokenizer=tokenizer, device=-1,
                    framework="pt", **options)


def load_transformer_pipeline(task, cache_directory=None, *, allow_download=False):
    """Load one pinned local pipeline. Network requires explicit opt-in.

    Downloads go directly to task/revision directories, avoiding Windows cache
    blob/snapshot copies. Only a single safetensors checkpoint is downloaded.
    Failed loads are not cached. Never loads remote code or uses paid inference.
    """
    directory = model_directory(task, cache_directory)
    spec = MODELS[task]
    try:
        if allow_download:
            from huggingface_hub import snapshot_download
            print(f"Explicit download: {spec['id']} at {spec['revision']} (public; no token).", flush=True)
            snapshot_download(spec["id"], revision=spec["revision"], local_dir=str(directory), token=False,
                              allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                              "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"])
            _load_local_pipeline.cache_clear()
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError("Pinned safetensors checkpoint is missing.")
        return _load_local_pipeline(task, str(directory))
    except Exception as error:
        raise TransformerModelUnavailable(
            f"Cannot load {spec['id']} from {directory}. Run python -m src.transformer_resources "
            f"--download --task {task} while online; then verify without --download. "
            "For offline use copy the complete .cache/transformer_tasks directory. "
            "Check requirements, disk space and permissions. Manual attention needs no model. "
            f"Setup detail: {type(error).__name__}: {error}"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--task", choices=["all", *MODELS], default="all")
    parser.add_argument("--cache-dir", type=Path, default=MODEL_CACHE)
    parser.add_argument("--report", type=Path, help="Optional measured instructor report")
    args = parser.parse_args()
    from src.transformers_nlp import analyze_sentiment, recognize_entities, answer_from_context
    records = []
    for task in MODELS if args.task == "all" else [args.task]:
        started = time.perf_counter()
        try:
            pipe = load_transformer_pipeline(task, args.cache_dir, allow_download=args.download)
            loaded = time.perf_counter()
            if task == "sentiment":
                output = [analyze_sentiment(pipe, text) for text in
                          ["The NLP workshop was engaging and easy to follow.",
                           "The instructions were confusing and frustrating."]]
            elif task == "ner":
                output = recognize_entities(pipe, "Riya studies Natural Language Processing at Hindu College of Engineering.")
            else:
                output = answer_from_context(pipe, "In which semester is Natural Language Processing offered?",
                                             "Natural Language Processing is offered in Semester 6. The course introduces text preprocessing, TF-IDF, word embeddings, attention, Transformers and semantic retrieval.")
            inference = time.perf_counter() - loaded
            repeat_start = time.perf_counter()
            assert load_transformer_pipeline(task, args.cache_dir) is pipe
            repeat_seconds = time.perf_counter() - repeat_start
        except TransformerModelUnavailable as error:
            parser.exit(1, str(error) + "\n")
        size = sum(path.stat().st_size for path in model_directory(task, args.cache_dir).rglob("*") if path.is_file())
        record = {"name": task, **MODELS[task], "cpu_load_or_download_seconds": loaded-started,
                  "cpu_inference_seconds": inference, "process_cached_load_seconds": repeat_seconds,
                  "cache_bytes": size, "output": output}
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)
    if args.report:
        import platform
        import importlib.metadata
        report = {"python": platform.python_version(), "download_requested": args.download,
                  "versions": {name: importlib.metadata.version(name) for name in ["transformers", "torch", "huggingface-hub"]},
                  "models": records, "total_cache_bytes": sum(row["cache_bytes"] for row in records)}
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
