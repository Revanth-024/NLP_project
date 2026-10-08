"""Exercise 11: mathematical alignment, provenance, caching and real evidence."""
from pathlib import Path
import copy
import hashlib
import json
import re
from unittest.mock import patch
import numpy as np
import pytest
from src.document_processor import load_markdown_document, process_document, model_token_lengths
from src.retrieval import (load_canonical_documents, build_chunk_corpus, encode_document_chunks,
    encode_user_question, calculate_chunk_similarities, retrieve_top_k_chunks, semantic_chunk_search)
from src.sentence_resources import load_sentence_embedding_model, SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION
ROOT=Path(__file__).resolve().parents[1]


class RecordingEncoder:
    """Test-only shape/ordering spy; never an application fallback."""
    def __init__(self):
        self.calls=[]
    def encode(self,texts,**settings):
        self.calls.append((texts,settings))
        matrix=np.zeros((len(texts),384),dtype=np.float32)
        for i,text in enumerate(texts):
            matrix[i,0]=len(text)
            matrix[i,1]=1
        return matrix


def tiny_chunks():
    document={'document_id':'D','title':'Title','source_type':'markdown','source_path':'example.md',
              'pages':[{'page_number':None,'text':'alpha beta gamma delta epsilon zeta'}]}
    return process_document(document,3,1)['chunks']


def test_text_only_encoding_alignment_and_no_mutation():
    chunks=tiny_chunks();before=copy.deepcopy(chunks);model=RecordingEncoder()
    matrix=encode_document_chunks(model,chunks)
    assert matrix.shape==(3,384) and np.isfinite(matrix).all()
    assert model.calls[0][0]==[c['text'] for c in chunks]
    assert matrix[:,0].tolist()==[len(c['text']) for c in chunks]
    assert chunks==before
    assert model.calls[0][1]['normalize_embeddings'] is False
    assert encode_document_chunks(model,[]).shape==(0,384)
    assert len(model.calls)==1


@pytest.mark.parametrize('invalid',['', '   ',None])
def test_blank_or_invalid_chunk_text(invalid):
    chunks=tiny_chunks();chunks[0]['text']=invalid
    with pytest.raises(ValueError): encode_document_chunks(RecordingEncoder(),chunks)


def test_question_shape_and_same_encoder_path():
    model=RecordingEncoder();chunks=tiny_chunks();encode_document_chunks(model,chunks)
    vector=encode_user_question(model,'alpha beta')
    assert len(vector)==384 and all(np.isfinite(vector))
    assert model.calls[-1][0]==['alpha beta']
    assert vector==encode_user_question(model,'alpha beta')


@pytest.mark.parametrize('question',['',' \n ',None,[]])
def test_bad_question(question):
    with pytest.raises((ValueError,TypeError)): encode_user_question(RecordingEncoder(),question)


def test_cosine_alignment_finite_zero_and_reuse():
    a=np.zeros(384);a[0]=1
    matrix=np.zeros((3,384));matrix[0,0]=2;matrix[1,1]=1
    with patch('src.retrieval.cosine_similarity',wraps=__import__('src.classical_nlp',fromlist=['cosine_similarity']).cosine_similarity) as spy:
        scores=calculate_chunk_similarities(a.tolist(),matrix)
        assert scores==pytest.approx([1,0,0]) and spy.call_count==3
    assert calculate_chunk_similarities(a.tolist(),np.empty((0,384)))==[]
    assert calculate_chunk_similarities(a.tolist(),matrix)==scores


@pytest.mark.parametrize('matrix',[np.zeros((2,383)),np.zeros(384),np.full((1,384),np.nan)])
def test_invalid_matrix(matrix):
    with pytest.raises(ValueError): calculate_chunk_similarities([0.0]*384,matrix)


def test_ranking_stable_ties_full_provenance_and_no_mutation():
    chunks=tiny_chunks();before=copy.deepcopy(chunks)
    ranked=retrieve_top_k_chunks(chunks,[.5,.8,.8],5)
    assert [c['chunk_id'] for c in ranked]==[chunks[1]['chunk_id'],chunks[2]['chunk_id'],chunks[0]['chunk_id']]
    assert [c['rank'] for c in ranked]==[1,2,3]
    for result in ranked:
        original=next(c for c in chunks if c['chunk_id']==result['chunk_id'])
        assert all(result[k]==v for k,v in original.items())
    assert chunks==before
    assert retrieve_top_k_chunks(chunks,[.5,.8,.8],0)==[]
    assert len(retrieve_top_k_chunks(chunks,[.5,.8,.8],1))==1
    assert retrieve_top_k_chunks([],[],5)==[]


@pytest.mark.parametrize('k',[-1,True,1.5,'5'])
def test_invalid_k(k):
    with pytest.raises(ValueError): retrieve_top_k_chunks(tiny_chunks(),[1,0,0],k)


def test_bad_provenance_duplicate_ids_and_lengths():
    chunks=tiny_chunks()
    with pytest.raises(ValueError): retrieve_top_k_chunks(chunks,[1,0],5)
    with pytest.raises(ValueError): retrieve_top_k_chunks([chunks[0],chunks[0]],[1,1],5)
    chunks[0]['page_number']=1
    with pytest.raises(ValueError): retrieve_top_k_chunks(chunks,[1,0,0],5)
    chunks[0]['source_type']='pdf'
    assert retrieve_top_k_chunks(chunks,[1,0,0],1)[0]['page_number']==1
    chunks[0]['page_number']=None
    with pytest.raises(ValueError): retrieve_top_k_chunks(chunks,[1,0,0],1)


def test_search_shape_mismatch_and_sorted_indices():
    model=RecordingEncoder();chunks=tiny_chunks();matrix=encode_document_chunks(model,chunks)
    with pytest.raises(ValueError): semantic_chunk_search('alpha',chunks,matrix[:-1],model)
    result=semantic_chunk_search('alpha',chunks,matrix,model,2)
    assert len(result['scores'])==len(chunks) and len(result['results'])==2
    for index,chunk in zip(result['sorted_indices'],retrieve_top_k_chunks(chunks,result['scores'],len(chunks))):
        assert chunks[index]['chunk_id']==chunk['chunk_id']
    assert semantic_chunk_search('alpha',[],np.empty((0,384)),model)['results']==[]


@pytest.fixture(scope='module')
def knowledge_base():
    documents=load_canonical_documents(ROOT);chunks=build_chunk_corpus(documents)
    model=load_sentence_embedding_model()
    lengths=model_token_lengths(model.tokenizer,[c['text'] for c in chunks])
    assert len(documents)==14 and len(chunks)==152 and max(lengths)==247
    assert not any(n>model.max_seq_length for n in lengths)
    matrix=encode_document_chunks(model,chunks)
    return chunks,matrix,model


def test_real_knowledge_base_shape_identity(knowledge_base):
    chunks,matrix,model=knowledge_base
    assert matrix.shape==(152,384) and np.isfinite(matrix).all()
    assert model.max_seq_length==256
    assert len({c['chunk_id'] for c in chunks})==152
    assert all(c['source_type']=='markdown' and c['page_number'] is None for c in chunks)
    one=encode_document_chunks(model,[chunks[50]])[0]
    assert np.allclose(matrix[50],one,atol=1e-5)




def test_real_similarity_semantic_sanity_and_unsupported(knowledge_base):
    chunks,matrix,model=knowledge_base
    from src.embeddings import encode_sentences
    vectors=encode_sentences(model,['Natural language processing analyzes human language.',
        'Computers process and understand human communication.','Computer networks route packets between devices.'])
    scores=calculate_chunk_similarities(vectors[0].tolist(),vectors)
    assert scores[0]==pytest.approx(1,abs=1e-5) and scores[1]>scores[2]
    result=semantic_chunk_search('What is the recipe for chocolate cake?',chunks,matrix,model,5)
    assert len(result['results'])==5 and all(np.isfinite(result['scores']))




