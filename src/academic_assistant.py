"""Final integration: deterministic routing over existing NLP capabilities.

No new student algorithms, model initialization, expected answers or UI imports.
Resource callbacks are invoked only by the capability that needs them.
"""
import re
import time
from src.question_analyzer import (
    UNMATCHED, normalize_question_text, match_question_topics,
    load_topic_taxonomy, load_question_papers, analyze_paper_collection,
    filter_questions_by_topic,
)
from src.rag import run_transparent_rag, select_context_chunks

MODES = {'auto', 'academic_qa', 'question_papers', 'study_support'}
CAPABILITY_NAMES = {
    'academic_qa': 'Academic Knowledge Q&A / Course Navigation',
    'question_papers': 'Previous Question Paper Analysis',
    'study_support': 'Study / Revision Support',
}
FREQUENCY_NOTE = ('These counts describe the four synthetic workshop papers and '
                  'do not predict future examination questions.')


def route_question(question, mode='auto'):
    """Papers first; strong study intent; academic rules/navigation; fallback.

    Weak 'study' never overrides a prerequisite/semester request. Explicit mode
    overrides automatic rules. Rules match phrase patterns, never stored queries.
    """
    if mode not in MODES:
        raise ValueError('Select Auto, Academic Q&A, Previous Papers or Study Support.')
    text = normalize_question_text(question)
    rules = []
    paper = r'\b(previous (?:nlp )?(?:papers?|questions?)|question papers?|asked before|frequent topics?|most frequent|how many(?: \w+){0,5} questions?|(?:show|list)(?: \w+){0,5} questions?)\b'
    study = r'\b(summariz\w*|revise|revision|explain (?:\w+ )?topics?|prepare (?:\w+ )?questions?)\b'
    academic = r'\b(attendance|examinations?|exam|semester|credits?|prerequisites?|curriculum|syllabus|calendar|laboratory|lab|placement|faculty|before|course|subjects?|marks|pass)\b'
    if re.search(paper, text) or re.search(r'\b(?:transformers?|attention) questions?\b', text):
        rules.append('paper_request')
    if re.search(study, text): rules.append('study_request')
    if re.search(academic, text): rules.append('academic_or_navigation')
    if re.search(r'\bstudy\b', text): rules.append('weak_study')
    if mode != 'auto':
        capability, reason = mode, 'Explicit mode selected by the user; automatic priority overridden.'
    elif 'paper_request' in rules:
        capability, reason = 'question_papers', 'Historical-paper/filter/frequency intent has priority over revision intent.'
    elif 'study_request' in rules:
        capability, reason = 'study_support', 'Summary or revision request will use retrieved academic evidence.'
    elif 'academic_or_navigation' in rules:
        capability, reason = 'academic_qa', 'Academic rules or course/curriculum navigation request.'
    elif 'weak_study' in rules:
        capability, reason = 'study_support', 'Study request will use retrieved academic evidence.'
    else:
        capability = 'academic_qa'
        reason = 'No specialized analyzer intent was detected, so this question is being handled through academic document retrieval.'
    return {'capability': capability, 'reason': reason, 'matched_rules': rules,
            'mode': mode, 'fallback': mode == 'auto' and not rules}


def load_analyzer_resources(root):
    """Read actual papers and the taxonomy projection, independently of models."""
    taxonomy = load_topic_taxonomy(root / 'data/university_facts.json')
    papers = load_question_papers(root / 'data/question_papers')
    return taxonomy, analyze_paper_collection(papers, taxonomy)


def parse_paper_request(question, taxonomy):
    """Reuse canonical alias matching; distinguish list/count/rank/comparison."""
    text = normalize_question_text(question)
    topic_match = match_question_topics(question, taxonomy)
    if re.search(r'\bcompare\b|\bacross papers\b', text): intent = 'paper_comparison'
    elif re.search(r'\bhow many\b|\bnumber of\b|\bcount\b', text): intent = 'topic_count'
    elif re.search(r'\bfrequen\w*\b|\bmost common\b', text): intent = 'topic_frequency'
    elif topic_match['primary_topic'] != UNMATCHED: intent = 'topic_filter'
    elif re.search(r'\b(revise|revision)\b', text) and re.search(r'\bprevious papers?\b', text):
        intent = 'topic_frequency'
    else: intent = 'unknown_topic'
    return {'intent': intent, 'topic_id': topic_match['primary_topic'],
            'topic_match': topic_match}


def inspect_unit_evidence(question, supplied_chunks):
    """Assistant-only lexical guard; never rerank or rewrite evidence.

    Resolve course names/acronyms from supplied syllabus titles, not an answer
    table. An explicit Unit identifier needs its heading in that course's
    supplied context. A heading is necessary, not proof of complete coverage.
    Ambiguous/missing course scope fails closed for optional generation.
    """
    requested = sorted(set(int(n) for n in re.findall(r'\bunit\s+(\d+)\b', question, re.I)))
    if not requested:
        return {'requested_units': [], 'generation_allowed': True, 'applied': False}
    names = sorted(set(re.sub(r'\s+syllabus$', '', c['document_title'], flags=re.I)
                       for c in supplied_chunks if c['document_title'].lower().endswith('syllabus')))
    scoped = []
    for name in names:
        acronym = ''.join(w[0] for w in name.split() if w.lower() not in {'and', 'for', 'of'})
        if re.search(r'\b' + re.escape(name) + r'\b', question, re.I) or (
                len(acronym) >= 2 and re.search(r'\b' + re.escape(acronym) + r'\b', question, re.I)):
            scoped.append(name)
    chunks = [c for c in supplied_chunks if len(scoped) == 1 and
              c['document_title'].lower() == (scoped[0] + ' Syllabus').lower()]
    found = sorted(set(int(n) for c in chunks for n in
                       re.findall(r'(?mi)^\s*#{1,6}\s+Unit\s+(\d+)\b', c['text'])))
    missing = [n for n in requested if n not in found]
    return {'applied': True, 'requested_units': requested, 'resolved_courses': scoped,
            'supplied_unit_headings': found, 'missing_units': missing,
            'generation_allowed': len(scoped) == 1 and not missing,
            'policy': 'Matching course and explicit unit heading required before generation; semantic scores/order unchanged.',
            'limitation': 'Heading presence does not establish complete or correct unit coverage.'}


def run_academic_assistant(question, mode='auto', *, retriever=None,
                           analyzer_loader=None, llm_loader=None,
                           generation_enabled=False, top_k=5):
    """One request, one capability. Optional generation cannot disable retrieval.

    retriever(question, K) follows the existing semantic_chunk_search API.
    analyzer_loader() returns (taxonomy, production collection analysis).
    llm_loader() is called only for requested academic/study generation.
    """
    started = time.perf_counter()
    routing = route_question(question, mode)
    route_seconds = time.perf_counter() - started
    result = {'question': question, 'capability': routing['capability'],
              'routing': routing, 'answer_type': 'unsupported', 'answer': None,
              'data': {}, 'sources': [], 'warnings': [],
              'diagnostics': {'routing_seconds': route_seconds}}
    if not question.strip():
        result['answer'] = 'Enter a non-empty question.'
        return result
    if routing['capability'] == 'question_papers':
        try:
            if analyzer_loader is None: raise ValueError('Question-paper data is unavailable.')
            taxonomy, analysis = analyzer_loader()
            request = parse_paper_request(question, taxonomy)
            intent, topic = request['intent'], request['topic_id']
            result['data'] = {'request': request, 'unmatched_count': analysis['unmatched_count'],
                              'unmatched_questions': [q for q in analysis['questions'] if q['primary_topic'] == UNMATCHED],
                              'provenance': 'historical/synthetic-paper'}
            result['warnings'].extend([FREQUENCY_NOTE,
                f"{analysis['unmatched_count']} questions remain unmatched by current deterministic taxonomy aliases; no model relabeling."])
            if intent == 'paper_comparison':
                result.update(answer_type=intent, answer='Observed primary-topic counts across papers.')
                result['data']['rows'] = analysis['per_paper']
            elif intent == 'topic_frequency':
                result.update(answer_type=intent, answer='Observed primary-topic ranking.')
                result['data']['rows'] = analysis['ranking']
            elif intent in {'topic_filter', 'topic_count'} and topic != UNMATCHED:
                questions = filter_questions_by_topic(analysis['questions'], topic, taxonomy)
                result.update(answer_type=intent,
                              answer=f'{len(questions)} primary-topic questions matched {topic}.')
                result['data'].update(questions=questions, count=len(questions), topic_id=topic)
                result['sources'] = [{'paper_id': q['paper_id'], 'question_id': q['question_id'],
                                      'source_path': q['source_path']} for q in questions]
            else:
                result['answer'] = 'No supported topic matched. Try a canonical topic such as Transformers, self-attention, CBOW or GPT.'
            if intent in {'paper_comparison', 'topic_frequency'}:
                result['sources'] = list({q['paper_id']: {'paper_id': q['paper_id'],
                    'source_path': q['source_path']} for q in analysis['questions']}.values())
            result['diagnostics']['analyzer_seconds'] = time.perf_counter() - started - route_seconds
        except (OSError, ValueError, KeyError) as error:
            result['answer'] = 'Previous-paper data is unavailable. Academic document retrieval remains independently available.'
            result['warnings'].append(str(error))
        return result

    try:
        if retriever is None: raise ValueError('No semantic retrieval resource is available.')
        retrieval_started = time.perf_counter()
        retrieved = retriever(question, top_k)
        retrieval_seconds = time.perf_counter() - retrieval_started
        supplied, _ = select_context_chunks(retrieved['results'])
        guard = inspect_unit_evidence(question, supplied)
        result['diagnostics'].update(unit_evidence=guard, original_query=question, retrieval_query=question)
        backend = None
        if not guard['generation_allowed']:
            result['warnings'].append('Requested unit evidence is missing or course scope is ambiguous. Generation withheld; inspect the syllabus and use its full course name. Retrieved evidence remains visible.')
        elif generation_enabled and llm_loader is not None:
            try: backend = llm_loader()
            except Exception as error:
                result['warnings'].append(f'Local generation unavailable: {error}')
        rag = run_transparent_rag(question, lambda q, k: retrieved, backend, top_k=top_k)
        rag['timings_seconds']['total'] += retrieval_seconds
        rag['timings_seconds']['retrieval'] = retrieval_seconds
    except Exception as error:
        result['answer'] = 'Semantic retrieval resources are unavailable. Previous-paper analysis remains available.'
        result['warnings'].append(str(error))
        return result
    result['data']['rag'] = rag
    if len(normalize_question_text(question).split()) <= 4:
        result['warnings'].append('Short queries can retrieve incomplete evidence. Try a complete question with the full course name; inspect all requirements rather than assuming the first result is sufficient.')
    result['sources'] = rag['attribution']['supplied_sources']
    result['diagnostics'].update(rag=rag['diagnostics'], timings_seconds=rag['timings_seconds'])
    result['warnings'].extend(rag['diagnostics']['flags'])
    result['warnings'].append(rag['attribution']['note'])
    result['answer'] = rag['generation']['answer']
    if result['answer'] is None:
        result['answer_type'] = 'retrieval_only'
        result['warnings'].append('Local generation is unavailable or not requested, but the retrieved academic evidence is shown below.')
    elif routing['capability'] == 'study_support':
        revision_questions = bool(re.search(r'\b(create|prepare|generate)\b.*\bquestions?\b', normalize_question_text(question)))
        result['answer_type'] = 'study_generation'
        result['data'].update(provenance='generated',
                              heading='AI-Generated Revision Questions' if revision_questions else 'AI-Generated Study Content')
        result['warnings'].append('Generated study material is not a historical examination record.')
    else:
        result['answer_type'] = 'rag_generated'
    if rag['generation']['error']: result['warnings'].append(rag['generation']['error'])
    result['warnings'].append('Similarity is not factual confidence; retrieved evidence and generated claims require inspection.')
    return result
