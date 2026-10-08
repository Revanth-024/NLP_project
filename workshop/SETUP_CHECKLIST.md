# Setup Checklist

## Recommended environment

64-bit Python3.11 is the workshop baseline; Python3.12 is also execution-tested. Install Git if cloning/committing. Obtain the owner's starter/reference repository URL; no public URL is invented here. After cloning or unpacking, work from the repository root.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is blocked by your institution, use `.\.venv\Scripts\python.exe` in place of `python`; do not change machine security settings just to activate a shell.

macOS/Linux:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Explicit online resource preload

Run before class, not on app reruns:

```text
python -m src.nltk_resources --download
python -m src.sentence_resources --download
python -m src.transformer_resources --download
```

NLTK installs stopwords/WordNet only; WordPunct/Porter need no extra tokenizer download. Optional POS demo: `python -m src.nltk_resources --download --include-pos`. No spaCy model, `punkt`, `omw-1.4`, GPU, paid service or Hugging Face token is required.

Optional capable-machine/instructor generation:

```text
python -m src.llm_resources --download
```

Qwen remains optional: approximately1GB cache, historical~2.12GB process RAM, multi-second CPU inference. Multiple loaded models add overhead; these are observations, not ceilings.

## Verify and launch

```text
python -m workshop.preflight
python -m workshop.preflight --full-classroom --smoke
python -m streamlit run app.py
```

PASS is inventory presence; WARNING identifies missing learning models in partial mode; strict `--full-classroom` makes them FAIL. Qwen is always OPTIONAL. A FAIL returns nonzero. Cache presence is not loadability; `--smoke` checks local inference. In the generated starter, unimplemented algorithm checks can report expected TODO WARNINGs; model/data failures still require repair.

Confirm home and all ten pages open. Reference pages compute results. Starter pages explain their first missing TODO; this is expected, not a broken installation.

## Tests

Reference:

```text
python -m pytest
python -m workshop.run_reliability_audit --pages
```

Student starter:

```text
python -m pytest tests/test_starter_smoke.py
python -m pytest tests/test_preprocessing.py
```

Exercise correctness tests are intentionally red before the cores are implemented. Use the Student Guide's per-exercise commands; do not run the complete reference suite after every small TODO. Optional notebook tools (`jupyter`, `nbformat`) are not app runtime dependencies; install in your notebook environment only if needed.

## Offline verification / recovery

After preload, disconnect network and verify without `--download`:

```text
python -m src.nltk_resources
python -m src.sentence_resources
python -m src.transformer_resources
python -m workshop.preflight --full-classroom --smoke
```

Copy complete `.nltk_data` and `.cache` trees to each prepared repository for offline class use, preserving pinned snapshots/files. Never commit them. Check disk space and permissions. The generated starter intentionally omits caches; its validator can borrow explicitly supplied instructor resources for validation only.

Missing Qwen preserves retrieval/context/papers. Missing MiniLM preserves deterministic paper analysis. Missing task models preserve manual attention. Missing paper data does not disable academic retrieval. Follow the displayed resource command, restore canonical files or recover only the affected exercise. All uploads stay in memory: text-based, unencrypted PDFs,10MiB/50pages, no OCR.

Cold model/kernel loads can take tens of seconds and vary by laptop; download/install time is excluded from the6½/7-hour teaching schedules. Every student should complete retrieval/prompt construction. Prefer instructor generation demos, not compulsory Qwen on all machines.
