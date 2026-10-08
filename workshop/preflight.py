"""Fast offline classroom inventory. Never downloads or loads model weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from src.sentence_resources import SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION
from src.transformer_resources import MODELS
from src.llm_resources import MODEL_REVISION
ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ['streamlit', 'pytest', 'nltk', 'scikit-learn', 'gensim', 'sentence-transformers', 'pypdf', 'torch', 'transformers']


def check_preflight(root=ROOT, full_classroom=False):
    """Cache presence is inventory, not loadability. --smoke verifies inference."""
    root = Path(root)
    checks = []
    def record(name, status, detail): checks.append({'check': name, 'status': status, 'detail': detail})
    supported = sys.version_info[:2] in {(3, 11), (3, 12)} and sys.maxsize > 2**32
    record('Python', 'PASS' if supported else 'FAIL', f'{platform.python_version()} / {platform.architecture()[0]}; workshop baseline 64-bit3.11, also validated3.12.')
    for package in PACKAGES:
        try: record(package, 'PASS', importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError: record(package, 'FAIL', 'Install requirements.txt in the active virtual environment.')
    try:
        facts = json.loads((root/'data/university_facts.json').read_text(encoding='utf-8'))
        manifest = json.loads((root/'data/document_manifest.json').read_text(encoding='utf-8'))
        paths = [root/'data'/d['filename'] for d in manifest['academic_documents']]
        good = facts['institution']['name'] == 'Hindu College of Engineering' and len(paths) == 14 and all(p.is_file() for p in paths)
        record('Academic data', 'PASS' if good else 'FAIL', 'Canonical facts /14 academic files; run dataset tests for content consistency.')
    except (OSError, ValueError, KeyError) as error: record('Academic data', 'FAIL', str(error))
    papers = sorted((root/'data/question_papers').glob('nlp_question_paper_*.md'))
    record('Question papers', 'PASS' if len(papers) == 4 else 'FAIL', f'{len(papers)}/4 actual Markdown papers; restore supplied data if missing.')
    corpus = root/'data/search_corpus/course_descriptions.csv'
    record('Course corpus', 'PASS' if corpus.is_file() else 'FAIL', str(corpus))
    try:
        from src.nltk_resources import resource_available
        for name in ['stopwords', 'wordnet']:
            available = resource_available(name)
            record('NLTK '+name, 'PASS' if available else 'FAIL', 'Verify python -m src.nltk_resources; explicit online setup adds --download.')
    except Exception as error: record('NLTK resources', 'FAIL', str(error))
    missing_status = 'FAIL' if full_classroom else 'WARNING'
    snapshot = root/'.cache/sentence_transformers'/('models--'+SENTENCE_MODEL_ID.replace('/','--'))/'snapshots'/SENTENCE_MODEL_REVISION
    required = ['model.safetensors','config.json','tokenizer.json','modules.json','1_Pooling/config.json']
    present = all((snapshot/n).is_file() for n in required)
    record('MiniLM cache', 'PASS' if present else missing_status,
           'Pinned local file inventory; preload: python -m src.sentence_resources --download. Verify: python -m src.sentence_resources.')
    for task, spec in MODELS.items():
        folder = root/'.cache/transformer_tasks'/task/spec['revision']
        present = all((folder/n).is_file() for n in ['model.safetensors','config.json','tokenizer_config.json'])
        record('Transformer '+task, 'PASS' if present else missing_status,
               f'Preload: python -m src.transformer_resources --download --task {task}. Missing task models leave manual attention usable.')
    folder = root/'.cache/local_llm'/MODEL_REVISION
    present = all((folder/n).is_file() for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json'])
    record('Qwen', 'OPTIONAL', 'Cache present; generation remains optional.' if present else 'Cache absent; retrieval/papers still work. Optional explicit setup: python -m src.llm_resources --download.')
    return {'python_version': platform.python_version(), 'full_classroom': full_classroom,
            'checks': checks, 'passed': not any(c['status']=='FAIL' for c in checks),
            'scope': 'No network or downloads. Cache inventory does not prove successful loading; use --smoke for local runtime checks.'}


def smoke_checks():
    """Optional local inference check, without Qwen; slower than inventory."""
    from src.preprocessing import tokenize_words
    from src.embeddings import prepare_tokenized_sentences, train_word2vec, encode_sentences
    from src.sentence_resources import load_sentence_embedding_model
    from src.transformer_resources import load_transformer_pipeline
    from src.transformers_nlp import analyze_sentiment, recognize_entities, answer_from_context
    checks=[]
    operations = {
        'Tokens': lambda: tokenize_words('Students learn NLP.'),
        'Word2Vec': lambda: train_word2vec(prepare_tokenized_sentences(['students learn language', 'students learn data']), epochs=5).wv.vector_size,
        'MiniLM inference': lambda: encode_sentences(load_sentence_embedding_model(), ['Students study NLP.']).shape,
        'Sentiment inference': lambda: analyze_sentiment(load_transformer_pipeline('sentiment'), 'The workshop is useful.'),
        'NER inference': lambda: recognize_entities(load_transformer_pipeline('ner'), 'Riya studies in London.'),
        'QA inference': lambda: answer_from_context(load_transformer_pipeline('qa'), 'What semester?', 'NLP is offered in Semester6.'),
    }
    for name, operation in operations.items():
        try: operation(); checks.append({'check': name, 'status': 'PASS', 'detail': 'Local operation succeeded.'})
        except NotImplementedError as error: checks.append({'check': name, 'status': 'WARNING', 'detail': str(error)})
        except Exception as error: checks.append({'check': name, 'status': 'FAIL', 'detail': str(error)})
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-classroom', action='store_true', help='Require all learning model caches; Qwen stays optional.')
    parser.add_argument('--smoke', action='store_true', help='Perform local model inference without downloading/Qwen.')
    args=parser.parse_args(); report=check_preflight(full_classroom=args.full_classroom)
    if args.smoke: report['checks'].extend(smoke_checks())
    for c in report['checks']: print(f"{c['status']:8} {c['check']}: {c['detail']}")
    print(report['scope'])
    return 1 if any(c['status']=='FAIL' for c in report['checks']) else 0

if __name__=='__main__': raise SystemExit(main())
