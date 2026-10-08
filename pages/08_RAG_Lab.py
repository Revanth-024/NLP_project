"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Exercise 14: one explicit RAG pipeline with optional local generation."""
    from pathlib import Path
    import hashlib
    import json
    import time
    import numpy as np
    import streamlit as st
    from src.document_processor import (load_markdown_document,extract_text_from_pdf,process_document,
        model_token_lengths,DocumentExtractionError)
    from src.retrieval import (load_canonical_documents,build_chunk_corpus,encode_document_chunks,semantic_chunk_search)
    from src.sentence_resources import (load_sentence_embedding_model,load_sentence_tokenizer,SentenceModelUnavailable,
        SENTENCE_MODEL_ID,SENTENCE_MODEL_REVISION)
    from src.llm_resources import local_model_status,load_local_llm,LocalModelUnavailable,MODEL_ID,MODEL_REVISION
    from src.rag import run_transparent_rag
    ROOT=Path(__file__).resolve().parents[1]


    @st.cache_data(show_spinner=False,max_entries=4)
    def cached_canonical_chunks(documents,chunk_size,overlap):
        return build_chunk_corpus(documents,chunk_size,overlap)


    @st.cache_resource(show_spinner='Encoding passages with cached CPU MiniLM...',max_entries=4)
    def cached_chunk_matrix(chunks,chunk_size,overlap,model_id,revision):
        if (model_id,revision)!=(SENTENCE_MODEL_ID,SENTENCE_MODEL_REVISION):
            raise ValueError('Use the pinned workshop encoder.')
        model=load_sentence_embedding_model()
        lengths=model_token_lengths(model.tokenizer,[c['text'] for c in chunks])
        if any(n>model.max_seq_length for n in lengths):
            raise ValueError('Some chunks exceed the encoder window. Reduce experimental chunk size; no silent truncation.')
        started=time.perf_counter();matrix=encode_document_chunks(model,chunks)
        return model,matrix,max(lengths,default=0),time.perf_counter()-started


    st.set_page_config(page_title='Transparent RAG Lab',page_icon='📚')
    st.title('Transparent RAG Lab')
    st.write('Exercise 14 — inspect a complete educational Retrieval-Augmented Generation pipeline. No new algorithm TODOs.')
    st.success('Input → Extraction → Cleaning → Chunking → Embeddings → Retrieval → Context → Prompt → Optional Local Generation → Transparent RAG')
    st.caption('Retrieval selects evidence. Context is what is supplied. Generation is model output. Attribution identifies supplied sources; it does not verify claims.')
    PRESETS={
        'Attendance':'What is the minimum attendance required?',
        'Examination marks':'What is the minimum end-semester score required to pass?',
        'NLP semester':'In which semester is Natural Language Processing offered?',
        'NLP prerequisites':'Which subjects must a student complete before taking NLP?',
        'NLP credits — indirect support':'How many credits does the NLP course carry?',
        'Multi-source attendance/examination':'When can low attendance stop a student from writing the final examination?',
        'Unsupported cake':'What is the recipe for chocolate cake?',
        'RET-06 curriculum':'Where do connected devices learn the rules for exchanging messages?',
        'ML / DL comparison':'Compare the focus of Machine Learning and Deep Learning.',
    }
    preset=st.selectbox('Question preset',list(PRESETS),key='rag_preset')
    question=st.text_input('Question',value=PRESETS[preset],key='rag_question_'+preset)
    mode=st.radio('Classroom mode',['Retrieval + Prompt Only','Full Local Generation'],horizontal=True,key='rag_mode')
    top_k=st.select_slider('Top-K evidence',options=[1,3,5,8],value=5,key='rag_k')

    with st.expander('Corpus, context budget and document-processing lab'):
        corpus=st.radio('Retrieval corpus',['Canonical 14-document knowledge base','Selected document / PDF experiment'],key='rag_corpus')
        max_chunks=st.select_slider('Maximum context chunks',options=[1,3,5,8],value=5,key='rag_context_chunks')
        max_chars=st.number_input('Maximum context characters',min_value=0,max_value=16000,value=8000,step=500,key='rag_context_chars')
        st.caption('Canonical retrieval stays 100 words / overlap 20. Other settings below apply only to a selected-source experiment.')
        manifest=json.loads((ROOT/'data/document_manifest.json').read_text(encoding='utf-8'))
        kind=st.radio('Inspect source',['Canonical Markdown','Sample PDF','Upload PDF'],horizontal=True,key='rag_inspect_kind')
        document=None
        try:
            if kind=='Upload PDF':
                upload=st.file_uploader('Text-based PDF (10 MiB / 50 pages maximum)',type=['pdf'])
                if upload is not None: document=extract_text_from_pdf(upload.getvalue(),source_name=upload.name)
                else: st.info('Choose a PDF to inspect. Canonical retrieval remains available.')
            else:
                entries=manifest['academic_documents' if kind=='Canonical Markdown' else 'derived_sample_pdfs']
                selected=st.selectbox('Document',range(len(entries)),format_func=lambda i:entries[i]['title'],key='rag_document_'+kind)
                entry=entries[selected]
                loader=load_markdown_document if kind=='Canonical Markdown' else extract_text_from_pdf
                document=loader(ROOT/'data'/entry['filename'],document_id=entry['document_id'],title=entry['title'])
        except DocumentExtractionError as error:
            st.warning(str(error))
        size=st.select_slider('Experimental chunk size (words)',options=[20,40,60,80,100,120,160,240],value=100,key='rag_size')
        overlap=st.select_slider('Experimental overlap (words)',options=[n for n in [0,10,20,40,60] if n<size],value=20 if size>20 else 0,key='rag_overlap')
        processed=None
        if document:
            processed=process_document(document,size,overlap)
            st.dataframe([processed['statistics']],hide_index=True)
            page=st.selectbox('Text unit',range(len(document['pages'])),format_func=lambda i:'Markdown — no physical page' if kind=='Canonical Markdown' else f'PDF page {i+1}',key='rag_page_'+kind)
            st.write('Raw extracted text');st.code(document['pages'][page]['text'],language=None,wrap_lines=True)
            st.write('Conservatively cleaned text');st.code(processed['cleaned_pages'][page]['text'],language=None,wrap_lines=True)
            if processed['chunks']:
                selected_chunk=st.selectbox('Inspect chunk',range(len(processed['chunks'])),format_func=lambda i:processed['chunks'][i]['chunk_id'],key='rag_inspect_chunk')
                chunk=processed['chunks'][selected_chunk]
                st.code(chunk['text'],language=None,wrap_lines=True)
                st.json({k:v for k,v in chunk.items() if k!='text'})
            if st.checkbox('Optional tokenizer-only length check',key='rag_token_check'):
                try:
                    tokenizer=load_sentence_tokenizer()
                    counts=model_token_lengths(tokenizer,[c['text'] for c in processed['chunks']])
                    st.write({'maximum_model_tokens':max(counts,default=0),'above_256':sum(n>256 for n in counts)})
                except SentenceModelUnavailable as error: st.info(str(error))
            st.caption('Offsets address cleaned text, end-exclusive. Markdown has no physical page. PDF windows never cross pages; uploaded bytes stay in memory, no OCR or persistence.')

    if not question.strip():
        st.session_state.pop('rag_result', None)
        st.session_state.pop('rag_result_identity', None)
        st.info('Enter a non-empty question.');st.stop()
    try:
        if corpus.startswith('Canonical'):
            sources=load_canonical_documents(ROOT)
            chunks=cached_canonical_chunks(sources,100,20)
            retrieval_size,retrieval_overlap=100,20
        else:
            if processed is None:
                st.info('Choose a usable selected document/PDF or return to canonical retrieval.');st.stop()
            chunks=processed['chunks'];retrieval_size,retrieval_overlap=size,overlap
        if not chunks:
            st.info('No non-empty chunks are available.');st.stop()
        embedder,matrix,token_max,encoding_seconds=cached_chunk_matrix(chunks,retrieval_size,retrieval_overlap,SENTENCE_MODEL_ID,SENTENCE_MODEL_REVISION)
    except (SentenceModelUnavailable,ValueError) as error:
        st.warning(str(error));st.stop()
    status=local_model_status()
    full=mode=='Full Local Generation'
    if full: st.info('Optional CPU generation can take several seconds. About 1 GB model cache; measured whole-process memory about 2.12 GB. Generated output can be incorrect.')
    if full and not status['files_present']:
        st.info('Generation is unavailable on this machine. You can still inspect retrieval, context and the grounded prompt. Optional explicit setup: python -m src.llm_resources --download')
    clicked=st.button('Generate and inspect RAG' if full else 'Retrieve and inspect RAG',key='rag_run')
    # Fingerprint every relevant input: an old answer must never accompany new evidence.
    identity=hashlib.sha256(json.dumps({'question':question,'k':top_k,'max_chunks':max_chunks,'max_chars':max_chars,
        'corpus':corpus,'chunks':chunks,'size':retrieval_size,'overlap':retrieval_overlap,'embedding_revision':SENTENCE_MODEL_REVISION,
        'generation_revision':MODEL_REVISION,'mode':mode},sort_keys=True).encode()).hexdigest()
    if clicked or st.session_state.get('rag_result_identity')!=identity:
        backend=None
        if clicked and full and status['files_present']:
            try:
                with st.spinner('Loading optional local CPU model...'): backend=load_local_llm()
            except LocalModelUnavailable as error: st.warning(str(error))
        with st.spinner('Retrieving evidence and preparing the pipeline...'):
            result=run_transparent_rag(question,lambda q,k:semantic_chunk_search(q,chunks,matrix,embedder,k),backend,
                top_k=top_k,max_chunks=max_chunks,max_characters=int(max_chars),generation_model=MODEL_ID if backend else None)
        st.session_state['rag_result']=result
        st.session_state['rag_result_identity']=identity
    result=st.session_state['rag_result']

    st.header('Generated Answer')
    generation=result['generation']
    if generation['status']=='generated':
        st.write(generation['answer'])
        st.caption('Local model output. Inspect evidence and relationships; this is not a verified academic answer.')
    elif generation['status']=='error': st.warning(generation['error'])
    else: st.info('Generation is unavailable or not requested. Retrieval, context and the grounded prompt are complete and inspectable.')

    st.header('Sources Supplied to the Model')
    st.caption('Deterministic attribution from included context, independent of model-emitted labels. In prompt-only mode these are the sources prepared for the model. This does not validate every generated claim.')
    for source in result['attribution']['supplied_sources']:
        page='N/A' if source['page_number'] is None else source['page_number']
        with st.expander(f"[{source['label']}] {source['document_title']} · {source['chunk_id']} · page {page} · retrieval rank {source['rank']}"):
            st.code(source['text'],language=None,wrap_lines=True)
            st.json({k:v for k,v in source.items() if k!='text'})
    if not result['attribution']['supplied_sources']: st.info('No whole source blocks fit this context budget.')

    with st.expander('Retrieved Evidence — exact Top-K text and provenance'):
        for item in result['retrieval']['results']:
            st.write(f"Rank {item['rank']} · Similarity {item['score']:.4f} · {item['document_title']} · {item['chunk_id']}")
            st.code(item['text'],language=None,wrap_lines=True)
            st.json({k:v for k,v in item.items() if k!='text'})
    with st.expander('Embeddings, similarity scores and sorted indices'):
        vector=np.asarray(result['retrieval']['query_embedding'])
        scores=result['retrieval']['scores']
        st.write({'chunk_matrix_shape':list(matrix.shape),'question_shape':list(vector.shape),'question_norm':float(np.linalg.norm(vector)),
            'first_8_dimensions':vector[:8].tolist(),'highest_similarity':max(scores),'lowest_similarity':min(scores),
            'mean_similarity':sum(scores)/len(scores),'sorted_indices':result['retrieval']['sorted_indices']})
        st.dataframe([{'index':i,'chunk_id':chunk['chunk_id'],'document':chunk['document_title'],'Similarity':scores[i]} for i,chunk in enumerate(chunks)],hide_index=True)
        st.caption('Similarity is not confidence. Query/Top-K changes reuse cached corpus embeddings; source/settings/revision changes invalidate the corresponding cache.')
    with st.expander('Context — exactly included and excluded evidence'):
        st.dataframe([result['context']['statistics']],hide_index=True)
        st.json(result['context']['excluded_sources'])
        st.code(result['context']['text'] or '[NO RETRIEVED EVIDENCE]',language=None,wrap_lines=True)
        st.caption('Whole-block rank prefix, 5 chunks / 8000 characters by default. No hidden merging/deduplication. Generation tokens differ from characters and MiniLM tokens.')
    with st.expander('Exact grounded prompt'):
        st.code(result['prompt'],language=None,wrap_lines=True)
    with st.expander('Model details and timing'):
        st.json(status)
        st.write({'embedding_model':SENTENCE_MODEL_ID,'embedding_revision':SENTENCE_MODEL_REVISION,'token_max':token_max,'initial_corpus_encoding_seconds':encoding_seconds})
        st.json(result['timings_seconds'])
        st.caption('Qwen stays CPU float32/greedy, one beam, 96 output tokens / 3072 input cap. About 1 GB cache and observed 2.12 GB whole-process RAM; timings/memory are observations, not guarantees. Opening this page never downloads or loads Qwen.')
    with st.expander('Diagnostics — structural observations, not correctness judgments',expanded=False):
        st.json(result['diagnostics'])
        st.write('Model-generated source labels:')
        st.json(result['attribution']['model_label_inspection'])
        st.caption('A valid label exists in the supplied mapping; it does not establish factual support. No labels is a separate attribution observation, not proof the answer is wrong.')
    with st.expander('Failure taxonomy and instructor experiments'):
        st.markdown('**Retrieval:** useful evidence missing from Top-K. **Context:** retrieved evidence excluded before prompting. **Generation:** supplied evidence misinterpreted. **Attribution:** missing/unknown labels. **Unsupported:** the knowledge base has no answer. Human inspection is required.')
        st.write('Compare attendance K=1/3/5 without changing prompt/model. Try a one-chunk context budget at K=5. Run cake, multi-source attendance/examination, NLP credits, RET-06 and comparison presets. Read all blocks: the credit syllabus can be missed while lab guidelines provide indirect four-credit support.')
        report_path=ROOT/'workshop/rag_evaluation_results.json'
        if report_path.exists():
            report=json.loads(report_path.read_text(encoding='utf-8'))
            st.caption('Recorded evaluator measurements only. They never replace the live pipeline output above.')
            st.json(report['summary'])
            st.dataframe([{'case':row['case_id'],'K':row['top_k'],'context_support':row['evidence_classification'],
                          'recorded_answer':row['answer'],'label_status':row['label_validity']['status']} for row in report['cases']],hide_index=True)
        st.write('Reproduce: python -m workshop.evaluate_rag (prompt-only), or add --generate after explicit model preload.')
    st.caption('Complete educational transparent RAG; not production-ready. Continue to the implemented Question Paper Analyzer and Academic Assistant pages.')

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
