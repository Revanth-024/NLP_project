"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Final assistant UI; independent lazy resources and caching are infrastructure."""
    from pathlib import Path
    import hashlib
    import json
    import streamlit as st
    from src.academic_assistant import run_academic_assistant, route_question, CAPABILITY_NAMES
    from src.question_analyzer import load_topic_taxonomy, extract_questions, analyze_paper_collection
    from src.retrieval import load_canonical_documents, build_chunk_corpus, encode_document_chunks, semantic_chunk_search
    from src.document_processor import model_token_lengths
    from src.sentence_resources import load_sentence_embedding_model, SENTENCE_MODEL_REVISION
    from src.llm_resources import load_local_llm, MODEL_REVISION
    ROOT = Path(__file__).resolve().parents[1]


    @st.cache_data(show_spinner=False, max_entries=2)
    def cached_papers(raw_papers, taxonomy):
        papers = [{'paper_id': extract_questions(text, path)[0]['paper_id'],
                   'questions': extract_questions(text, path)} for path, text in raw_papers]
        return taxonomy, analyze_paper_collection(papers, taxonomy)


    def analyzer_loader():
        taxonomy = load_topic_taxonomy(ROOT / 'data/university_facts.json')
        files = sorted((ROOT / 'data/question_papers').glob('nlp_question_paper_*.md'))
        if not files: raise ValueError('No synthetic question papers found.')
        return cached_papers([(f.as_posix(), f.read_text(encoding='utf-8-sig')) for f in files], taxonomy)


    @st.cache_resource(show_spinner='Preparing local academic evidence...', max_entries=2)
    def cached_retrieval(documents, revision):
        if revision != SENTENCE_MODEL_REVISION: raise ValueError('Use the pinned workshop encoder revision.')
        model = load_sentence_embedding_model()
        chunks = build_chunk_corpus(documents, 100, 20)
        if any(n > model.max_seq_length for n in model_token_lengths(model.tokenizer, [c['text'] for c in chunks])):
            raise ValueError('Academic chunks exceed the encoder window; no silent truncation.')
        return model, chunks, encode_document_chunks(model, chunks)


    def retrieve(question, k):
        model, chunks, matrix = cached_retrieval(load_canonical_documents(ROOT), SENTENCE_MODEL_REVISION)
        return semantic_chunk_search(question, chunks, matrix, model, top_k=k)


    st.set_page_config(page_title='Intelligent Academic Assistant', page_icon='📚')
    st.title('Intelligent Academic Assistant')
    st.write('Ask about the fictional Hindu College of Engineering, explore previous papers, or revise from academic evidence.')
    st.caption('Academic Q&A • Course / Curriculum Navigation • Previous Papers • Grounded Study Support')
    MODES = {'Auto': 'auto', 'Academic Q&A': 'academic_qa', 'Previous Papers': 'question_papers', 'Study Support': 'study_support'}
    mode = st.radio('Mode', list(MODES), horizontal=True)
    PRESETS = {
        'Academic Rules': ['What is the minimum attendance required?', 'What marks are required to pass?', 'When is the NLP end-semester examination?'],
        'Courses': ['In which semester is NLP offered?', 'What are the prerequisites for NLP?', 'Which courses should I study before Deep Learning?', 'Compare Machine Learning and Deep Learning.'],
        'Previous Papers': ['Show Transformer questions from previous papers.', 'Show Attention questions.', 'How many Word2Vec questions appeared?', 'What are the most frequent topics in previous NLP papers?', 'Compare topics across previous papers.'],
        'Study Support': ['Summarize NLP Unit 4.', 'What should I revise for Attention?', 'Create five revision questions for NLP Unit 4.'],
    }
    group = st.selectbox('Example group', list(PRESETS))
    example = st.selectbox('Example question', PRESETS[group])
    question = st.text_input('Your question', value=example, key='assistant_question_' + example)
    generation = st.checkbox('Enable optional local generation', value=False)
    if generation: st.info('Optional CPU generation can take several seconds. About 1 GB model cache; observed whole-process memory about 2.12 GB. Output is not guaranteed correct.')
    with st.expander('Evidence settings'):
        top_k = st.select_slider('Top-K evidence', options=[1, 3, 5, 8], value=5)
        st.caption('Canonical 100-word chunks, overlap 20; at most five chunks / 8,000 characters enter context. No automatic model download.')
    route = route_question(question, MODES[mode])
    st.subheader('Routing')
    st.write('Detected capability: **' + CAPABILITY_NAMES[route['capability']] + '**')
    st.write(route['reason'])
    identity = hashlib.sha256(json.dumps([question, mode, generation, top_k, SENTENCE_MODEL_REVISION, MODEL_REVISION]).encode()).hexdigest()
    if st.session_state.get('assistant_identity') != identity:
        st.session_state.pop('assistant_result', None)
        st.session_state['assistant_identity'] = identity
    if st.button('Ask assistant', type='primary'):
        with st.spinner('Running the selected capability...'):
            st.session_state['assistant_result'] = run_academic_assistant(
                question, MODES[mode], retriever=retrieve, analyzer_loader=analyzer_loader,
                llm_loader=load_local_llm, generation_enabled=generation, top_k=top_k)
    result = st.session_state.get('assistant_result')
    if result:
        kind, data = result['answer_type'], result['data']
        if kind in {'rag_generated', 'study_generation'}:
            st.subheader(data.get('heading', 'Generated Answer'))
            st.write(result['answer'])
        elif kind == 'retrieval_only':
            st.subheader('Retrieved Evidence')
            st.info('Generation withheld: the requested unit heading is missing or course scope is ambiguous.' if not result['diagnostics'].get('unit_evidence', {}).get('generation_allowed', True) else 'Local generation is unavailable or not requested. Expand Retrieved Sources below; no answer has been fabricated.')
        else:
            st.subheader('Previous-Paper Questions' if kind == 'topic_filter' else 'Result')
            st.write(result['answer'])
        if kind == 'topic_filter':
            st.dataframe([{'Paper': q['paper_id'], 'Question': q['question_id'], 'Marks': q['marks'], 'Text': q['text']} for q in data['questions']], hide_index=True)
        if kind == 'topic_count': st.metric('Observed primary-topic questions', data['count'])
        if kind == 'topic_frequency':
            st.dataframe(data['rows'][:5], hide_index=True)
            with st.expander('Full observed ranking'): st.dataframe(data['rows'], hide_index=True)
        if kind == 'paper_comparison': st.dataframe(data['rows'], hide_index=True)
        if data.get('provenance') == 'historical/synthetic-paper':
            with st.expander('Unmatched historical questions'):
                st.dataframe([{'Paper': q['paper_id'], 'Question': q['question_id'], 'Text': q['text']}
                              for q in data['unmatched_questions']], hide_index=True)
            with st.expander('Historical source provenance'): st.json(result['sources'])
        if 'rag' in data:
            rag = data['rag']
            st.subheader('Sources Supplied to the Model' if result['answer'] else 'Retrieved Sources')
            st.caption('Supplied evidence, not verified citations. Markdown sources have no physical page numbers.')
            for source in result['sources']:
                with st.expander(f"{source['label']} — {source['document_title']} ({source['chunk_id']})", expanded=False):
                    st.write(source['text'])
                    st.caption(f"Similarity {source['score']:.4f} • {source['source_path']}")
            with st.expander('Retrieved evidence / prompt-ready context'):
                st.dataframe([{'Rank': c['rank'], 'Document': c['document_title'], 'Chunk': c['chunk_id'], 'Score': c['score']} for c in rag['retrieval']['results']], hide_index=True)
                st.code(rag['context']['text'], language=None, wrap_lines=True)
                st.code(rag['prompt'], language=None, wrap_lines=True)
        for warning in result['warnings']: st.warning(warning)
        with st.expander('Diagnostics'):
            st.json({'routing': result['routing'], 'answer_type': kind, **result['diagnostics']})
    st.caption('Counts describe synthetic historical papers. Optional Qwen can produce incorrect or incomplete answers, even with relevant evidence.')
    st.page_link('pages/08_RAG_Lab.py', label='Inspect the full transparent RAG workflow')
    st.page_link('pages/09_Question_Paper_Analyzer.py', label='Inspect deterministic question analysis')

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
