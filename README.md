# nlp-academic-assistant-student

This NLP repo is for students.

# NLP-Based Intelligent Academic Assistant — Student Starter

A 2-Day Hands-On NLP Workshop: **From Classical NLP to Generative AI: Building an Intelligent Academic Assistant**.

This generated starter contains 49 intended algorithm TODOs, shared `src` APIs, ten educational pages, eight notebooks, and fictional academic data. It is not the completed reference implementation. Infrastructure and model loading are provided; complete the NLP cores in `src/`.

Begin with [START_HERE](START_HERE.md), [Student Guide](docs/STUDENT_GUIDE.md), [Setup Checklist](workshop/SETUP_CHECKLIST.md), and [Schedule](workshop/WORKSHOP_SCHEDULE.md).

Hindu College of Engineering is fictional. **Synthetic educational data created for NLP workshop purposes.**

Read [Known Limitations](docs/KNOWN_LIMITATIONS.md), [Deployment Guide](docs/DEPLOYMENT_GUIDE.md), and [Portfolio Guide](docs/GITHUB_PORTFOLIO_GUIDE.md). Model caches are not bundled; setup is explicit and free, with Qwen optional. Repository license awaits owner selection; no legal license was silently assigned.

```text
python -m workshop.preflight
python -m streamlit run app.py
python -m pytest tests/test_starter_smoke.py
python -m pytest tests/test_preprocessing.py
```

The last command intentionally fails until its TODOs are complete; do not treat incomplete exercises as broken infrastructure. Formula/expected-output tests validate your implementation, not a hidden answer engine. No instructor solutions or measured development reports are included.
