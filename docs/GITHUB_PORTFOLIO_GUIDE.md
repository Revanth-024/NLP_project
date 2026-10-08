# GitHub Portfolio Guide

## Describe your work honestly

Suggested repository description:

> Hands-on NLP project progressing from classical text processing and TF-IDF to semantic search, Transformers and transparent RAG using open-source local models.

Suggested topics: `nlp`, `python`, `streamlit`, `tf-idf`, `word2vec`, `sentence-transformers`, `semantic-search`, `transformers`, `rag`, `education`.

## Customize the README

State which TODOs you implemented, the workshop context, your experiments and what remains incomplete. Replace owner placeholders only with your actual repository/demo URLs. Keep the fictional-data statement and exact disclaimer. Explain one lexical failure, one retrieval/generation failure and one deterministic-analysis choice. Do not present reference results or instructor metrics as your personal achievements.

## Screenshot checklist

Capture your own completed app: preprocessing stages, TF-IDF search, lexical/semantic comparison, attention calculations, Transformer task, RAG evidence/context/output, paper analyzer and final assistant. Show real inputs/results; label retrieval-only vs generated modes. Include source/limitation context. Never fabricate screenshots, citations, accuracy, user counts or deployment claims.

## Demo and résumé templates

Demo description: “I implemented [completed components] in a two-day NLP workshop, compared lexical and semantic retrieval on fictional academic data, and inspected grounding failures before integrating an assistant.”

Possible résumé bullets (customize to actual work):

- Implemented manual text representations, TF-IDF and cosine ranking with exercise-level tests.
- Compared Word2Vec and pretrained sentence representations, then integrated semantic retrieval with inspectable source context.
- Built deterministic topic/frequency analysis over synthetic papers and an explainable academic-assistant interface.

Do not add quantified achievements without your own measured evidence. In an interview, explain UI→orchestration→shared src→data/local models; distinguish retrieval from generation and explain why source display does not certify claims.

## Owner publication workflow (not executed automatically)

1. Select a repository license and review upstream dependency/model terms.
2. Validate reference and generate/validate the starter. Prefer a restricted instructor repository while exercises are in progress, and a separate generated starter/template for students; regenerate rather than independently editing two copies.
3. Create the remote repository yourself. Inspect `git status` and ignore rules; exclude weights/caches/secrets. Add your own description/screenshots.
4. Commit the intended source. If needed, initialize a Git repository yourself with `git init`; connect your actual remote with `git remote add origin <OWNER-SELECTED-URL>` and publish only when ready with `git push -u origin main`. Replace the placeholder; no remote URL exists in this package.
5. Optionally create a future `v1.0-workshop` tag/release after review. This task does not create it.

Student journey: clone generated starter→implement→test→experiment→commit→next exercise. Follow the commit journey in the schedule/Student Guide. Publish your completed work when your own checks and license decision are ready.
