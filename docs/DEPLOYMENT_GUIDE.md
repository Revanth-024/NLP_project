# Deployment Guide

**Deployment configuration validated locally; live cloud deployment not performed.** No push, public app creation or deployment occurs in this task.

## Local full workshop — validated baseline

Use64-bit Python3.11 (3.12 also tested), install requirements, explicitly preload resources, run preflight and start `python -m streamlit run app.py`. All models infer locally on CPU. Qwen is optional; all students can complete retrieval/context/prompt without it. Run the reference release for a completed demo, or a student repository after implementing its TODOs.

## Streamlit Community Cloud — owner preparation

Community Cloud takes a GitHub repository/branch and an entrypoint. Choose `app.py` at the repository root and explicitly select Python3.11 in Advanced settings;3.12 is also tested. No secrets are required here. Follow the [official deploy instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Keep the existing `requirements.txt` beside `app.py` and data/pages/src paths relative to the repository. Community Cloud installs Python requirements; its host runs Linux. No `packages.txt`, Docker or external system dependency is required by this project. See [official dependency setup](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Lightweight hosted behavior

“Lightweight” means **no large model inference**, not a small dependency download: the unchanged scientific stack still includes Transformer/PyTorch infrastructure. Installation/memory feasibility on free hosting is not demonstrated. Start with classical representations/search, local tiny Word2Vec, document preparation and deterministic paper analysis where their prerequisites are satisfied.

Model caches and `.nltk_data` are intentionally not committed or bundled. Normal execution does not download them. Missing MiniLM/task models give setup guidance; missing MiniLM leaves paper analysis usable. Missing Qwen leaves retrieval/context/prompt usable **if MiniLM is available**. Without MiniLM, hosted academic retrieval is unavailable rather than faked. Without NLTK corpora, affected preprocessing stages report missing resources.

An owner who wants neural hosted features must establish a supported, explicit resource-preload process for that host and validate it separately. The local setup commands are not automatically executed by Community Cloud. This release intentionally adds no surprise startup downloader/cloud bootstrap. Do not commit model binaries or assume an interactive shell/persistent cache is available on free hosting. The fully preloaded local workshop is the reliable choice.

## Qwen and limits

Do not require Qwen for hosted deployment. Its~1GB cache, float32 memory and CPU latency can exceed practical hosting resources. Keep generation disabled and use retrieval/analyzer behavior where available. Inspect platform logs/resource failures and follow [official app-management guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app); current limits are provider-controlled and may change.

## Local configuration review

Validated: Python-compatible syntax, root requirements/entrypoint, relative path construction, no hardcoded development-machine paths in code/docs/notebooks/starter, no required credentials, local-only loaders, missing-model behavior, in-memory upload guards and starter incomplete-state handling. Local validation does not certify Linux wheels, hosting quotas, resource persistence or a live deployment.

Before you publish: choose a repository license, check upstream licenses, complete TODOs if deploying the student project, run preflight/tests, keep caches/secrets ignored, use genuine screenshots, and test each hosted capability independently. Report unavailable neural features honestly; do not advertise a live RAG demo before verifying it.
