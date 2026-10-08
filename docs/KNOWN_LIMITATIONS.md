# Known Limitations

## Retrieval

TF-IDF depends on lexical overlap; dense similarity can recognize paraphrases but can still choose the wrong source. Short NLP prerequisite wording ranked complete evidence8th–10th; a full course-name question ranked it2nd. This is measured wording sensitivity, not a guarantee that longer queries work.

Unit identifiers are weak semantic signals: NLP Unit1–5 heading-bearing chunks ranked3/7/1/12/7. The assistant does not alter scores or ranking. Explicit `Unit n` requests require one matching syllabus course scope and matching supplied unit headings before generation; missing/ambiguous evidence withholds generation. This literal structural guard is conservative, not complete language interpretation or proof that the whole unit is supplied. Top-K always returns nearest chunks; high similarity is not factual confidence.

## Generation

Qwen2.5-0.5B-Instruct is optional CPU inference. ~1GB cache and historical~2.12GB process RAM are observations, not ceilings. Warm generation commonly takes seconds; observed attendance generation was~12.35s. Multiple loaded models add memory/cold-load costs.

Reasoning, multi-source comparison, unsupported questions and source-label emission can fail. Earlier Unit4 summary/revision output described other units; the assistant now withholds these measured missing-evidence requests. Pure RAG Lab remains a transparent demonstration of the underlying failure. AttendanceK1 can omit75% evidence. Retrieved/supplied sources are not automatically verified citations, and valid labels do not prove claim support.

## Analyzer

Over four synthetic papers:48 questions,39 correct assignments,9 unmatched,0 matched-but-wrong;81.25% closed-set accuracy and1.0 precision on emitted assignments. No hidden model relabeling. Aliases cannot capture every implicit paraphrase. Primary frequencies count each slot once; historical frequency does not predict future examinations.

## Documents

Text-based, unencrypted PDFs only:10MiB/50-page guards, memory-only uploads, no OCR. Word windows can split ideas/headings. Markdown has no physical page number. Data is fictional and cannot be used as real institutional policy.

## Deployment / UI

Local preloaded resources are the validated baseline. Free hosting may not fit the scientific dependency footprint or inference memory; Qwen is not required for lightweight deployment. Without cached models, neural features report unavailable, while classical/Word2Vec/paper/data features remain available as their prerequisites allow. NLTK resources also require explicit setup.

1280px browser width is verified; attempted1024px overrides did not apply in the in-app browser. Narrower responsive layouts remain a release-review item. This educational app makes no production security, privacy isolation or scalability claim.
