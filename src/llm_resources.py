"""Optional pinned CPU generator: explicit preload; normal use is local-only."""
import argparse
from functools import lru_cache
from pathlib import Path
import time
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
MODEL_LICENSE='Apache-2.0'
MODEL_CACHE=Path(__file__).resolve().parents[1]/'.cache/local_llm'
MAX_NEW_TOKENS=96
MAX_INPUT_TOKENS=3072


class LocalModelUnavailable(RuntimeError):
    """Optional generation setup failure, never a reason to disable retrieval."""


def model_directory(cache_directory=None):
    return Path(cache_directory or MODEL_CACHE).resolve()/MODEL_REVISION


def local_model_status(cache_directory=None):
    """Cheap file-presence status, not a guarantee of successful load/generation."""
    folder=model_directory(cache_directory)
    required=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json']
    return {'backend':'Transformers/PyTorch CPU','model_id':MODEL_ID,'revision':MODEL_REVISION,
            'license':MODEL_LICENSE,'files_present':all((folder/name).is_file() for name in required),
            'directory':str(folder),'inference':'local CPU; no hosted API'}


@lru_cache(maxsize=1)
def _load_local_backend(directory):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import torch
    from src.local_llm import TransformersLocalLLM
    torch.set_num_threads(min(4,torch.get_num_threads()))
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,dtype=torch.float32).to('cpu')
    model.eval()
    # Neutralize checkpoint sampling defaults; this workshop uses greedy decoding.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    return TransformersLocalLLM(model,tokenizer,max_new_tokens=MAX_NEW_TOKENS,max_input_tokens=MAX_INPUT_TOKENS)


def load_local_llm(cache_directory=None,*,allow_download=False):
    folder=model_directory(cache_directory)
    try:
        if allow_download:
            from huggingface_hub import snapshot_download
            print(f'Explicit public download: {MODEL_ID} at {MODEL_REVISION}; no token required.',flush=True)
            snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_dir=str(folder),token=False,
                allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json',
                                'tokenizer_config.json','special_tokens_map.json','README.md','LICENSE','merges.txt','vocab.json'])
            _load_local_backend.cache_clear()
        if not local_model_status(cache_directory)['files_present']:
            raise FileNotFoundError('Pinned local checkpoint/tokenizer files are missing.')
        return _load_local_backend(str(folder))
    except Exception as error:
        raise LocalModelUnavailable('Local generation model is not installed or cannot load. Retrieval and prompt construction remain available. '
            'Optional setup: python -m src.llm_resources --download, then python -m src.llm_resources. '
            'For offline use copy the complete .cache/local_llm directory. '
            f'Detail: {type(error).__name__}: {error}') from error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=MODEL_CACHE)
    args=parser.parse_args()
    start=time.perf_counter()
    try:
        backend=load_local_llm(args.cache_dir,allow_download=args.download)
        loaded=time.perf_counter()
        print(backend.generate('Answer briefly: What is two plus two?'))
    except LocalModelUnavailable as error:
        parser.exit(1,str(error)+'\n')
    print(f'CPU load {loaded-start:.3f}s; generation {time.perf_counter()-loaded:.3f}s')
    print(f'Cache bytes: {sum(p.stat().st_size for p in model_directory(args.cache_dir).rglob("*") if p.is_file())}')


if __name__=='__main__':
    main()
