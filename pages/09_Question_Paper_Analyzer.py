"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Deterministic paper analyzer UI; no model or ground-truth-derived results."""
    from pathlib import Path
    import json
    import streamlit as st
    from src.question_analyzer import (load_topic_taxonomy,load_question_papers,analyze_paper_collection,
                                      filter_questions_by_topic,UNMATCHED,SPECIFICITY)
    ROOT=Path(__file__).resolve().parents[1]
    st.set_page_config(page_title='Question Paper Analyzer',page_icon='📄')
    st.title('Deterministic NLP Question Paper Analyzer')
    st.write('Exercise 15: paper text → extracted questions → normalization → explicit topic/alias matches → primary frequencies → ranking → filtering.')
    st.info('Question frequency in these synthetic papers is a learning aid and does not predict future examination questions.')
    st.caption('Fictional Hindu College of Engineering. Synthetic educational data created for NLP workshop purposes. No embeddings, Transformer or LLM inference is used here.')
    try:
        taxonomy=load_topic_taxonomy(ROOT/'data/university_facts.json')
        papers=load_question_papers(ROOT/'data/question_papers')
        analyzed=analyze_paper_collection(papers,taxonomy)
    except (OSError,ValueError) as error:
        st.warning(f'Cannot analyze the prepared papers: {error}');st.stop()
    names={t['id']:t['name'] for t in taxonomy};names[UNMATCHED]='Unmatched'
    questions=analyzed['questions']
    st.subheader('Paper overview')
    st.dataframe([{'paper':p['paper_id'],'questions':len(p['questions']),'marks':sum(q['marks'] for q in p['questions'])} for p in papers],hide_index=True)
    st.write({'papers':len(papers),'top_level_questions':len(questions),'canonical_topics':len(taxonomy),
              'unmatched':analyzed['unmatched_count'],'ambiguous':analyzed['ambiguous_count']})
    with st.expander('Extraction and canonical taxonomy'):
        selected_paper=st.selectbox('Paper', [p['paper_id'] for p in papers],key='analyzer_paper')
        paper=next(p for p in papers if p['paper_id']==selected_paper)
        st.code(paper['raw_text'],language=None,wrap_lines=True)
        st.json(taxonomy)
        st.caption('Only ## Qn — m marks headings define top-level slots. Aliases are lexical matches; related topic IDs/subtopics never propagate assignments. GPT belongs to Language Models; there is no separate RAG category.')

    st.subheader('Question explorer — why this topic?')
    paper_id=st.selectbox('Explore paper',[p['paper_id'] for p in papers],key='analyzer_explore_paper')
    paper_questions=[q for q in questions if q['paper_id']==paper_id]
    question_id=st.selectbox('Question',[q['question_id'] for q in paper_questions],key='analyzer_question_'+paper_id)
    question=next(q for q in paper_questions if q['question_id']==question_id)
    st.write('Original');st.code(question['text'],language=None,wrap_lines=True)
    st.write('Normalized');st.code(question['normalized_text'],language=None,wrap_lines=True)
    st.write({'primary_topic':names[question['primary_topic']],'primary_id':question['primary_topic'],
              'reason':question['reason'],'ambiguous_candidates':question['ambiguous_candidates'],
              'paper_id':question['paper_id'],'question_id':question['question_id'],'marks':question['marks'],'source_path':question['source_path']})
    st.dataframe(question['matches'],hide_index=True)
    with st.expander('Matching policy and custom experiment'):
        st.write('NFKC/casefold, punctuation/hyphen boundaries, whitespace collapse and joined TFIDF normalization. No stemming or stopword removal. A limited final-word singular/plural variant makes Transformer/Transformers and representation/representations match.')
        st.json(SPECIFICITY)
        st.caption('Narrow explicit concepts first; then longer alias in words; then canonical taxonomy order. Equal-specificity/length candidates are flagged. UNMATCHED is a status, not a seventeenth topic. Phrase offsets refer to normalized text.')
        custom=st.text_input('Try a question',value='Explain multi-head self-attention in Transformer models.',key='analyzer_custom')
        from src.question_analyzer import match_question_topics
        st.json(match_question_topics(custom,taxonomy))

    st.subheader('Observed topic frequencies and ranking')
    st.dataframe(analyzed['ranking'],hide_index=True)
    st.bar_chart([{'Topic':row['topic_name'],'Questions':row['count']} for row in analyzed['ranking']],x='Topic',y='Questions')
    st.caption('Canonical topics include zero-count rows. Unmatched questions are reported separately. Each slot counts only once toward its primary topic; secondary matches are not added to totals.')
    st.write('Most frequent observed topic: '+analyzed['ranking'][0]['topic_name']+' — '+str(analyzed['ranking'][0]['count'])+' questions.')

    st.subheader('Filter questions by topic')
    topic=st.selectbox('Topic',[t['id'] for t in taxonomy]+[UNMATCHED],format_func=lambda id:names[id],index=next(i for i,t in enumerate(taxonomy) if t['id']=='transformers'),key='analyzer_topic')
    secondary=st.checkbox('Include explicit secondary matches',value=False,key='analyzer_secondary')
    filtered=filter_questions_by_topic(questions,topic,taxonomy,include_secondary=secondary)
    search=st.text_input('Optional text search in filtered questions',key='analyzer_search')
    if search.strip(): filtered=[q for q in filtered if search.casefold() in q['text'].casefold()]
    st.write(f'{len(filtered)} questions — '+('primary or explicit secondary matches' if secondary else 'primary-only matches'))
    for q in filtered:
        with st.expander(f"{q['paper_id']} / {q['question_id']} — primary {names[q['primary_topic']]}"):
            st.write(q['text']);st.json({'paper_id':q['paper_id'],'question_id':q['question_id'],
              'primary_topic':q['primary_topic'],'matched_phrase':q['primary_match']['matched_phrase'] if q['primary_match'] else None,
              'matches':q['matches'],'source_path':q['source_path']})
    st.subheader('Compare papers — observed primary counts')
    st.dataframe(analyzed['per_paper'],hide_index=True)
    with st.expander('Ambiguous and unmatched questions'):
        for q in questions:
            if q['primary_topic']==UNMATCHED or q['is_ambiguous']:
                st.write(f"{q['paper_id']} / {q['question_id']}: {q['text']}");st.write(q['reason'])

    if st.checkbox('Show evaluation / instructor analysis',value=False,key='analyzer_evaluation'):
        report_file=ROOT/'workshop/question_analyzer_results.json'
        if report_file.exists():
            report=json.loads(report_file.read_text(encoding='utf-8'))
            st.subheader('Observed Analyzer Result vs Known Synthetic Ground Truth')
            st.caption('Recorded evaluator output only. This optional section never supplies production classifications or frequency counts.')
            st.write({'correct':report['correct'],'incorrect':report['incorrect'],'unmatched':report['unmatched'],'accuracy':report['accuracy']})
            st.dataframe([{'topic':names[t['id']],'observed':report['observed_frequencies'][t['id']],
                 'ground_truth':report['ground_truth_frequencies'][t['id']],'difference':report['frequency_differences'][t['id']]} for t in taxonomy],hide_index=True)
            st.dataframe(report['per_topic_metrics'],hide_index=True)
            st.json(report['confusion_pairs'])
            st.write('Reproduce: python -m workshop.evaluate_question_analyzer')
        else: st.info('Run the evaluator to create its separate measured artifact.')
    st.subheader('Why deterministic NLP here?')
    st.write('Structured headings and explicit technical vocabulary make a small reproducible rule pipeline useful. It exposes every decision and needs no model download. Implicit paraphrases can remain unmatched; use that limitation to discuss when semantic models might help.')
    st.caption('Next: open the Academic Assistant to integrate deterministic paper analysis with academic retrieval. Historical counts remain separate from generated study content.')

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
