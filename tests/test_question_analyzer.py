from pathlib import Path
import copy
import hashlib
import json
from unittest.mock import patch
import pytest
from src.question_analyzer import *
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def taxonomy(): return load_topic_taxonomy(ROOT/'data/university_facts.json')


@pytest.fixture(scope='module')
def collection(taxonomy):
    papers=load_question_papers(ROOT/'data/question_papers')
    return papers,analyze_paper_collection(papers,taxonomy)


def test_actual_extraction(collection):
    papers,result=collection
    assert [p['paper_id'] for p in papers]==['NLP-P01','NLP-P02','NLP-P03','NLP-P04']
    assert len(result['questions'])==48
    for paper in papers:
        assert len(paper['questions'])==12
        assert [q['question_id'] for q in paper['questions']]==[f'Q{i}' for i in range(1,13)]
        assert sum(q['marks'] for q in paper['questions'])==60
        assert all(q['text'] and Path(q['source_path']).exists() for q in paper['questions'])
        assert not any('Paper ID:' in q['text'] or 'Answer all twelve' in q['text'] for q in paper['questions'])


def test_multiline_top_level_slots_and_bad_headers():
    text='Paper ID: P1\n## Q1 — 5 marks\nExplain NLP.\n### (a) A subpart\nContinuation.\n## Q2 — 5 marks\nExplain BERT.'
    records=extract_questions(text,'p.md')
    assert len(records)==2 and 'Continuation' in records[0]['text']
    assert records[0]['question_number']==1 and records[0]['source_path']=='p.md'
    for bad in ['text without ID','Paper ID: P1\nNo headings',text.replace('Q2','Q1'),text.replace('Q2','Q3'), 'Paper ID: P\n## Q1 — 5 marks\n']:
        with pytest.raises(ValueError): extract_questions(bad)


@pytest.mark.parametrize('text,expected',[
 ('Explain Multi-Head Self-Attention.','explain multi head self attention'),
 (' TF‑IDF\t\nTF IDF; TFIDF! ','tf idf tf idf tf idf'),
 ('Word2Vec, CBOW, Skip–Gram, BERT and GPT.','word2vec cbow skip gram bert and gpt'),
 ('N-Grams / n_grams','n grams n grams'),('',''),('  \n ','')])
def test_normalization(text,expected):
    assert normalize_question_text(text)==expected


@pytest.mark.parametrize('text,topic',[
 ('TF-IDF','tf_idf'),('term weighting','tf_idf'),('TF IDF','tf_idf'),
 ('word embeddings','word_embeddings'),('dense word representations','word_embeddings'),
 ('word-vector geometry','word_embeddings'),('Word2Vec','word2vec'),('CBOW','cbow'),
 ('Continuous Bag of Words','cbow'),('Skip-Gram','skip_gram'),
 ('attention mechanism','attention'),('self-attention','attention'),
 ('multi-head self-attention in Transformer models','attention'),
 ('Transformer architecture','transformers'),('Transformer','transformers'),
 ('BERT encoder','bert'),('Bidirectional Encoder Representations from Transformers','bert'),
 ('GPT','language_models'),('Generative Pre-trained Transformer','language_models'),
 ('meaning-based retrieval','semantic_search'),('word segmentation','tokenization')])
def test_topic_names_aliases_and_specificity(text,topic,taxonomy):
    match=match_question_topics(text,taxonomy)
    assert match['primary_topic']==topic and match['primary_match']['matched_phrase']
    assert match['primary_match']['match_type'] in ['alias','canonical']




def test_ambiguity_deterministic_and_no_identity_input(taxonomy):
    a=match_question_topics('Compare CBOW and BERT.',taxonomy)
    assert a['is_ambiguous'] and set(a['ambiguous_candidates'])=={'cbow','bert'}
    assert a['primary_topic']=='cbow'
    assert a==match_question_topics('Compare CBOW and BERT.',taxonomy)
    records=[{'paper_id':'A','question_id':'Q1','text':'self-attention in a Transformer'},
             {'paper_id':'B','question_id':'Q99','text':'self-attention in a Transformer'}]
    assigned=analyze_questions(records,taxonomy)
    assert assigned[0]['primary_topic']==assigned[1]['primary_topic']=='attention'


def test_primary_frequencies_ranking_zeroes_and_filters(taxonomy):
    questions=analyze_questions([{'paper_id':'P','question_id':f'Q{i}','text':text} for i,text in enumerate(['CBOW','self-attention and Transformer','Unrelated','CBOW'],1)],taxonomy)
    counts=count_topic_frequencies(questions,taxonomy)
    assert counts['cbow']==2 and counts['attention']==1 and counts['transformers']==0 and counts[UNMATCHED]==1
    assert sum(counts.values())==4
    ranking=rank_topics(counts,taxonomy)
    assert ranking[0]['topic_id']=='cbow' and len(ranking)==16
    assert all(r['topic_id']!=UNMATCHED for r in ranking)
    assert rank_topics({},taxonomy)==[]
    tied=rank_topics({'word2vec':1,'tf_idf':1},taxonomy)
    assert [r['topic_id'] for r in tied]==['tf_idf','word2vec']
    assert filter_questions_by_topic(questions,'transformers',taxonomy)==[]
    secondary=filter_questions_by_topic(questions,'transformers',taxonomy,include_secondary=True)
    assert len(secondary)==1 and secondary[0]['question_id']=='Q2'
    filtered=filter_questions_by_topic(questions,'cbow',taxonomy)
    assert len(filtered)==2 and filtered[0]['paper_id']=='P'
    assert len(filter_questions_by_topic(questions,UNMATCHED,taxonomy))==1
    with pytest.raises(ValueError): filter_questions_by_topic(questions,'unknown',taxonomy)
    with pytest.raises(ValueError): rank_topics({'cbow':-1},taxonomy)


def test_actual_transformer_filter_per_paper_and_integrity(collection,taxonomy):
    papers,result=collection
    filtered=filter_questions_by_topic(result['questions'],'transformers',taxonomy)
    assert [(q['paper_id'],q['question_id']) for q in filtered]==[(f'NLP-P0{i}',q) for i in range(1,5) for q in ['Q1','Q7']]
    for row in result['per_paper']:
        assert row['total']==sum(row[p['paper_id']] for p in papers)
    assert sum(row['total'] for row in result['per_paper'])==48
    assert len(result['ranking'])==16


def test_production_without_metadata_or_expected_counts(tmp_path,collection,taxonomy):
    papers,expected=collection
    for paper in papers:
        original=Path(paper['source_path'])
        (tmp_path/original.name).write_text(paper['raw_text'],encoding='utf-8')
    (tmp_path/'question_metadata.json').write_text('not valid JSON',encoding='utf-8')
    contract=tmp_path/'facts.json'
    contract.write_text(json.dumps({'nlp_topic_taxonomy':taxonomy}),encoding='utf-8')
    actual=analyze_paper_collection(load_question_papers(tmp_path),load_topic_taxonomy(contract))
    assert actual['frequencies']==expected['frequencies']
    assert [q['primary_topic'] for q in actual['questions']]==[q['primary_topic'] for q in expected['questions']]
    real_read=Path.read_text
    def guarded(path,*args,**kwargs):
        if path.name in ['question_metadata.json','question_analyzer_results.json']: raise AssertionError('Ground truth read by production')
        return real_read(path,*args,**kwargs)
    with patch.object(Path,'read_text',guarded):
        repeat=analyze_paper_collection(load_question_papers(ROOT/'data/question_papers'),load_topic_taxonomy(ROOT/'data/university_facts.json'))
        assert repeat['frequencies']==expected['frequencies']
    code=(ROOT/'src/question_analyzer.py').read_text(encoding='utf-8')
    assert 'question_metadata.json' not in code and 'question_paper_design' not in code
    assert 'streamlit' not in code.lower() and 'transformers import' not in code




