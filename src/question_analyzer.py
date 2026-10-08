"""Exercise 15: deterministic paper-text analysis; five reference TODO cores.

Only standard Python. Production inputs are Markdown question text and canonical
names/aliases. Ground-truth assignments/target counts belong to evaluation only.
"""
import json
from pathlib import Path
import re
import unicodedata
UNMATCHED='UNMATCHED'
# Lexical specificity, not additional topics/aliases or automatic related matches.
# Narrow architecture/objective distinctions are retained before broad mentions.
SPECIFICITY={'cbow':2,'skip_gram':2,'bert':2,'word2vec':1,'attention':1}


def load_topic_taxonomy(facts_path):
    """Project ONLY the authorized taxonomy fields from the canonical contract."""
    facts=json.loads(Path(facts_path).read_text(encoding='utf-8'))
    return [{'id':topic['id'],'name':topic['name'],'aliases':list(topic.get('aliases',[]))}
            for topic in facts['nlp_topic_taxonomy']]


def extract_questions(paper_text,source_path=''):
    """Starter parser for ## Qn — m marks; each entire slot is one record.

    Header/instructions are excluded. Continuation lines/subparts remain within
    their top-level slot. Reject empty/malformed/duplicate/nonsequential slots.
    IDs come from the actual Paper ID header, never evaluation metadata.
    """
    if not isinstance(paper_text,str): raise TypeError('Paper text must be a string.')
    identity=re.search(r'^Paper ID:\s*([A-Za-z0-9_-]+)\s*$',paper_text,flags=re.M)
    if not identity: raise ValueError('Missing Paper ID header.')
    headers=list(re.finditer(r'^##\s+Q(\d+)\s*[—–-]\s*(\d+)\s+marks?\s*$',paper_text,flags=re.M))
    if not headers: raise ValueError('No supported top-level question headings found.')
    numbers=[int(h.group(1)) for h in headers]
    if numbers!=list(range(1,len(headers)+1)):
        raise ValueError('Question IDs must be unique and sequential from Q1.')
    records=[]
    for index,header in enumerate(headers):
        end=headers[index+1].start() if index+1<len(headers) else len(paper_text)
        text=paper_text[header.end():end].strip()
        marks=int(header.group(2))
        if not text or marks<=0: raise ValueError('Question slots need text and positive marks.')
        records.append({'paper_id':identity.group(1),'question_id':f'Q{numbers[index]}',
                        'question_number':numbers[index],'marks':marks,'text':text,
                        'source_path':str(source_path),'source_char_start':header.end(), 'source_char_end':end})
    return records


def load_question_papers(directory):
    """Discover only actual paper Markdown; no metadata/count contract input."""
    files=sorted(Path(directory).glob('nlp_question_paper_*.md'))
    if not files: raise ValueError('No synthetic NLP Markdown question papers found.')
    papers=[]
    for file in files:
        text=file.read_text(encoding='utf-8-sig')
        records=extract_questions(text,file.as_posix())
        papers.append({'paper_id':records[0]['paper_id'],'source_path':file.as_posix(),
                       'raw_text':text,'questions':records})
    if len({p['paper_id'] for p in papers})!=len(papers): raise ValueError('Paper IDs must be unique.')
    return papers


def normalize_question_text(text):
    """Casefold/NFKC, separator normalization, whitespace; no stemming/stopwords.

    Technical alphanumerics survive: Word2Vec, CBOW, BERT, GPT. Hyphens and
    punctuation become word boundaries, so TF-IDF and tf idf are equivalent.
    TFIDF's joined orthographic form is separated explicitly. Offsets reported
    by matching address normalized text, not original paper bytes.
    """
    if not isinstance(text,str): raise TypeError('Question text must be a string.')
    # STUDENT TODO 14.1 — Normalize Question Text
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: consistent phrase boundaries while retaining technical words.
    # Hint: casefold, replace separators, then collapse whitespace.
    # BEGIN STUDENT CORE 14.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 14.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 14.1


def compile_topic_rules(taxonomy):
    """Starter phrase preparation: exact aliases, with a limited final-word s.

    English singular/plural surface variant only, not stemming/semantic matching.
    No related_topic_ids/subtopics become aliases. Canonical names win duplicate
    normalized forms (e.g. bag-of-words), avoiding artificial duplicate evidence.
    """
    if len({t['id'] for t in taxonomy})!=len(taxonomy): raise ValueError('Taxonomy IDs must be unique.')
    rules=[]
    for order,topic in enumerate(taxonomy):
        seen=set()
        for match_type,phrase in [('canonical',topic['name'])]+[('alias',a) for a in topic.get('aliases',[])]:
            normalized=normalize_question_text(phrase)
            if not normalized or normalized in seen: continue
            seen.add(normalized)
            words=normalized.split()
            last=words[-1]
            if len(last)>3:
                base=last[:-1] if last.endswith('s') else last
                tail=re.escape(base)+r's?'
            else: tail=re.escape(last)
            pattern=r'(?<!\w)'+(r'\s+'.join(re.escape(w) for w in words[:-1])+r'\s+' if len(words)>1 else '')+tail+r'(?!\w)'
            rules.append({'topic_id':topic['id'],'topic_name':topic['name'],'alias_phrase':phrase,
                'normalized_alias':normalized,'match_type':match_type,'phrase_words':len(words),
                'specificity':SPECIFICITY.get(topic['id'],0),'taxonomy_order':order,'pattern':pattern})
    return rules


def match_question_topics(text,taxonomy):
    """Explicit lexical matches only; explain primary/secondary/ambiguity.

    Priority: specificity tier, phrase word count, taxonomy order, occurrence,
    canonical-before-alias for equal duplicates. Identity is never an input.
    """
    normalized=normalize_question_text(text)
    rules=compile_topic_rules(taxonomy)
    # STUDENT TODO 14.2 — Match Questions to Topics and Synonyms
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: collect explicit phrase matches and choose a deterministic primary.
    # Hint: word boundaries, narrow topics first, longer phrase, taxonomy ties.
    # BEGIN STUDENT CORE 14.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 14.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 14.2
    topics=list(dict.fromkeys(m['topic_id'] for m in ordered))
    tied=list(dict.fromkeys(m['topic_id'] for m in ordered if primary and
                           (m['specificity'],m['phrase_words'])==(primary['specificity'],primary['phrase_words'])))
    return {'normalized_text':normalized,'primary_topic':primary['topic_id'] if primary else UNMATCHED,
        'primary_match':primary,'matched_topics':topics,'matches':ordered,'ambiguous_candidates':tied if len(tied)>1 else [],
        'is_ambiguous':len(tied)>1,
        'reason':('No canonical name/alias matched; not forced into a topic.' if primary is None else
                  f"Selected {primary['topic_id']}: specificity {primary['specificity']}, {primary['phrase_words']} phrase words; taxonomy order breaks equally specific/long matches. Related topics are not propagated.")}


def analyze_questions(records,taxonomy):
    """Starter composition preserves original question/paper provenance."""
    return [{**record,**match_question_topics(record['text'],taxonomy)} for record in records]


def count_topic_frequencies(questions,taxonomy):
    """One primary per record; all canonical topics plus UNMATCHED, including 0."""
    counts={topic['id']:0 for topic in taxonomy};counts[UNMATCHED]=0
    if any(q.get('primary_topic') not in counts for q in questions): raise ValueError('Unknown primary topic assignment.')
    # STUDENT TODO 14.3 — Count Topic Frequencies
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: each top-level slot increments only its selected primary topic.
    # Hint: initialize counts, then increment the primary key once per record.
    # BEGIN STUDENT CORE 14.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 14.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 14.3


def rank_topics(counts,taxonomy):
    """Descending observed count, canonical taxonomy order ties. UNMATCHED is reported separately.

    Zero counts are retained. Empty counts returns []. Canonical order only
    resolves tied frequency, never changes counts. Unknown IDs/negative counts fail.
    """
    order={t['id']:i for i,t in enumerate(taxonomy)};order[UNMATCHED]=len(order)
    names={t['id']:t['name'] for t in taxonomy};names[UNMATCHED]='Unmatched'
    if any(key not in order or isinstance(value,bool) or not isinstance(value,int) or value<0 for key,value in counts.items()):
        raise ValueError('Counts need known topic IDs and non-negative integers.')
    # STUDENT TODO 14.4 — Rank Topics by Frequency
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: most frequent first, deterministic ties.
    # Hint: negative count sorts descending; taxonomy position breaks ties.
    # BEGIN STUDENT CORE 14.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 14.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 14.4


def filter_questions_by_topic(questions,topic_id,taxonomy,*,include_secondary=False):
    """Copied records with full provenance; primary-only unless explicitly asked."""
    if topic_id not in {t['id'] for t in taxonomy}|{UNMATCHED}: raise ValueError('Select a canonical topic or UNMATCHED.')
    # STUDENT TODO 14.5 — Filter Questions by Topic
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: select matching records without stripping IDs/text/explanations.
    # Hint: primary equality; optional explicit secondary-match membership.
    # BEGIN STUDENT CORE 14.5
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 14.5: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 14.5


def analyze_paper_collection(papers,taxonomy):
    """Starter reporting: primary counts/ranks, per-paper matrix and flags."""
    records=[q for paper in papers for q in paper['questions']]
    questions=analyze_questions(records,taxonomy)
    counts=count_topic_frequencies(questions,taxonomy)
    matrix=[]
    for topic in rank_topics(counts,taxonomy)+[{'topic_id':UNMATCHED,'topic_name':'Unmatched','count':counts[UNMATCHED]}]:
        row={'topic_id':topic['topic_id'],'topic_name':topic['topic_name'],'total':topic['count']}
        for paper in papers:
            row[paper['paper_id']]=sum(q['paper_id']==paper['paper_id'] and q['primary_topic']==topic['topic_id'] for q in questions)
        matrix.append(row)
    return {'questions':questions,'frequencies':counts,'ranking':rank_topics(counts,taxonomy),'per_paper':matrix,
            'unmatched_count':counts[UNMATCHED],'ambiguous_count':sum(q['is_ambiguous'] for q in questions)}
