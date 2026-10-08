# Start Here

Build NLP components progressively: Words → Numbers → Meaning → Retrieval → Context → Generation → Intelligent Application.

1. After cloning/unpacking, use64-bit Python3.11 (3.12 also tested). Create/activate `.venv` and install `requirements.txt`; follow workshop/SETUP_CHECKLIST.md for platform commands.
2. Explicitly preload NLTK, MiniLM and task models before class. No paid credentials. Qwen is optional.
3. Run `python -m workshop.preflight`; run `python -m pytest tests/test_starter_smoke.py`.
4. First exercise: `src/preprocessing.py`, **STUDENT TODO1.1**. Preserve signatures/markers and change only the marked core.
5. Run `python -m streamlit run app.py`. Pages catch expected NotImplementedError messages, not unrelated coding bugs.
6. Run `python -m pytest tests/test_preprocessing.py`; later commands are in docs/STUDENT_GUIDE.md. Correctness tests fail while cores are incomplete—this is expected.
7. Experiment in notebook01, inspect each stage, then `git add src/preprocessing.py` and `git commit -m "Add text preprocessing"`. Continue through the schedule and commit journey.

After fixing an incomplete notebook TODO, restart the kernel and run cells again. Some later exercises depend on earlier ones; follow the first displayed TODO instead of changing infrastructure. Ask the instructor for a bounded recovery checkpoint when stuck.
