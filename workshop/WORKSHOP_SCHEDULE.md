# Two-Day Workshop Schedule

Pre-class: install dependencies, explicitly preload resources, run preflight/smoke and prepare the generated starter. Setup/downloads are not hands-on algorithm time. Qwen is optional and mainly an instructor demo.

## Day1 — Words → Numbers → Search → Embeddings → Meaning (6½ hours)

| Time | Block | Deliverable / TODOs |
|---|---|---|
|09:00–09:15|Opening and preflight|Project story, setup smoke, Checkpoint0 |
|09:15–10:00|Preprocessing|Visible stages;1.1–1.7; Checkpoint1 |
|10:00–10:45|One-hot / BoW / N-Grams|Shared coordinates;2.1–2.4; Checkpoint2 |
|10:45–10:55|Break| |
|10:55–12:15|TF-IDF / cosine / search|Hand calculations and engine;3.1–3.5 /4.1–4.3; Checkpoint3 |
|12:15–13:00|Lunch| |
|13:00–13:40|Word2Vec|CBOW/Skip-Gram;5.1–5.5; word-vector milestone |
|13:40–14:15|Sentence embeddings|Batch shapes/cosine;6.1–6.2 |
|14:15–14:25|Break| |
|14:25–15:05|Semantic search|Fair lexical comparison;7.1–7.4; Checkpoint4 |
|15:05–15:30|Experiment / commit / recovery buffer|A semantic search engine; explain one failure |

## Day2 — Transformers → Documents → Retrieval → RAG → Analysis → Assistant (7 hours)

| Time | Block | Deliverable / TODOs |
|---|---|---|
|09:00–09:10|Recap / warm resources|Review representations and context |
|09:10–09:50|Attention|Scores→softmax→context;8.1–8.3 |
|09:50–10:25|Transformer NLP experiments|Provided sentiment/NER/QA; no new algorithm TODO; Checkpoint5 |
|10:25–10:35|Break| |
|10:35–11:20|Documents / cleaning / chunking|10.1–10.4; private-knowledge preparation |
|11:20–12:00|Semantic chunk retrieval|11.1–11.4; Checkpoint6 |
|12:00–12:45|Lunch| |
|12:45–13:15|Context / grounded prompt|12.1–12.3; source provenance |
|13:15–13:50|Transparent RAG / optional Qwen demo|No new algorithm TODO; Checkpoint7; inspect failures |
|13:50–14:00|Break| |
|14:00–14:45|Deterministic paper analyzer|14.1–14.5; observed counts, not predictions |
|14:45–15:15|Assistant integration|Reuse components; no new algorithm TODO; Checkpoint8 |
|15:15–15:45|GitHub / deployment planning|Personal evidence and honest hosting scope; work toward Checkpoint9 |
|15:45–16:00|Final recap / commit / recovery|Architecture explanation and remaining limitations |

These are planning estimates. The49 cores need instructor pacing, particularly TF-IDF/search. Cold resource loading may take tens of seconds; Task17 notebook blocks took about90s in a shared prepared process; the Task18 RAG notebook alone took about265s under cold/competing validation load, including optional generation. Independent cold kernels and student laptops can differ. Do not wait for repeated generation from every laptop.

## Suggested student commit journey

| Completed block | Suggested commit |
|---|---|
|Preprocessing|Add text preprocessing |
|BoW / N-Grams|Add bag of words representations |
|TF-IDF / cosine|Build TF-IDF search |
|Word2Vec|Add word embedding experiments |
|Sentence similarity|Add sentence similarity |
|Semantic search|Build semantic search |
|Attention / task experiments|Add attention and transformer NLP |
|Document preparation|Process academic documents |
|Chunk retrieval|Add semantic chunk retrieval |
|Context / prompt|Build grounded RAG context |
|RAG integration|Complete transparent RAG pipeline |
|Paper analyzer|Build question paper analyzer |
|Final integration|Integrate academic assistant |

Use `git add <YOUR-CHANGED-FILES>` and `git commit -m "<MESSAGE>"` after your tests/experiments. Replace placeholders with actual paths/messages. Do not commit model caches, environments or private uploads.

