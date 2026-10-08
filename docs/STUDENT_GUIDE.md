> **Generated student starter:** implementations are intentionally incomplete. Expected outputs below describe completed exercises; reference-state history does not mean your TODOs are already solved. Follow START_HERE.md.

# Welcome

Welcome to **From Classical NLP to Generative AI: Building an Intelligent Academic Assistant**, a 2-day hands-on workshop for engineering students. Begin with the setup checklist. This repository provides the application shell, synthetic dataset and working reference Preprocessing, BoW/N-Grams, TF-IDF Search, Word2Vec, Sentence Similarity, Semantic Search and Transformer labs. Document processing, transparent RAG, the deterministic analyzer and the integrated assistant are now functional in reference form.

# What You Will Build

**NLP-Based Intelligent Academic Assistant Using Semantic Search, Transformers and Retrieval-Augmented Generation**. You will build this incrementally, observing how text becomes vectors, search results and grounded answers.

# How This Workshop Works

Clone → Set up environment → Verify installation → Run Streamlit → Learn → Find TODO → Implement → Test → Experiment → Observe → Commit → Continue → Integrate → Deploy.

Primarily edit `src/`. Read each exercise's expected inputs and outputs, implement its numbered TODO, run the relevant tests, then inspect intermediate results in its notebook or lab. Commit small working steps. There are no algorithm exercises to implement in Task 01.

# Repository Overview

- `src/`: reusable NLP implementations you will primarily edit.
- `pages/`: starter-provided Streamlit lab layouts and navigation pages.
- `notebooks/`: concept exploration; import `src/` rather than copy algorithms.
- `data/`: canonical fictional university facts, academic documents, course descriptions, question papers and beginner text examples.
- `tests/`: independent checks for your implementations; foundation, dataset consistency, preprocessing, representations, weighting, retrieval and Word2Vec checks.
- `docs/`: student instructions and instructor teaching notes.
- `workshop/`: setup checklist and evolving TODO registry.
- `assets/`: future architecture images and screenshots.
- `app.py`: Streamlit home page.

# Our Fictional University Dataset

The workshop uses **Hindu College of Engineering**, a completely fictional educational institution. Its synthetic academic details are intentionally consistent and will be reused across search, RAG, academic assistance and question-paper analysis labs. Later exercises will search and analyze the supplied documents; you are not expected to create the dataset yourself.

Task 02 establishes `data/university_facts.json` as the source of truth. Task 03 now supplies the documents and question papers. This is preparation for future labs and introduces no student NLP TODO. Every academic document explicitly identifies the institution as fictional and includes **Synthetic educational data created for NLP workshop purposes.**

## Meet the Workshop Dataset

You will work with the same fictional university across the workshop, so you can see how different methods handle familiar information. You are not expected to write this dataset yourself. Inspect the readable inputs now; algorithms will be added in later exercises.

- **Course descriptions → TF-IDF Search → Semantic Search.** Open `data/search_corpus/course_descriptions.csv` to inspect 20 short, contrasting course profiles.
- **Full academic documents → Chunking → Retrieval → RAG → Academic Assistant.** Browse `data/academic_documents/` for the 14 Markdown syllabi, policies, curriculum and calendar records.
- **Question papers → NLP processing → Topic analysis → Frequency ranking.** Browse the four `nlp_question_paper_*.md` files in `data/question_papers/`. They are undated synthetic previous-paper sets, each with twelve five-mark questions.

Small text inputs are available in `data/examples/` for preprocessing, word-count, N-Gram and similarity experiments. They contain inputs rather than worked answers. The document manifest and retrieval benchmarks help instructors and tests track source coverage; they are evaluation artifacts, not application answers.

Question metadata and target frequencies are testing ground truth. Your future analyzer must derive questions, topics and counts independently from actual paper text. Retrieval and RAG must use document content rather than expected benchmark answers. Tests may use these ground-truth artifacts to evaluate your implementation. Task 03 introduces no new student TODO.

# Student TODO Convention

Use `STUDENT TODO <module>.<exercise>` identifiers, for example `STUDENT TODO 3.2`. IDs are stable once assigned; later tasks assign IDs when exercises are designed.

- ★ Guided — scaffolded practice with direct hints.
- ★★ Core — essential independent implementation.
- ★★★ Challenge — optional deeper exploration.

Convention example using the now-assigned TODO 3.3 (its complete reference is in src):

```python
# ============================================================
# STUDENT TODO 3.3 — Calculate Inverse Document Frequency
# Difficulty: ★★ Core
# ============================================================
# Goal:
# Calculate how informative each vocabulary term is across
# the document collection.
#
# Formula:
# IDF(t) = ln(N / DF(t))  # natural logarithm
#
# Expected:
# Return IDF values aligned to the shared vocabulary order.
#
# Hint:
# Terms appearing in fewer documents should receive
# larger IDF values.
# ============================================================
```

Include Goal, Formula where relevant, Expected and Hint. Explain return values and meaningful edge cases when the exercise is designed. Task 01 contains no assigned algorithm TODOs.

# Architecture and Learning Principles

```text
Streamlit UI
     ↓
src modules
     ↓
NLP logic
```

Pages may import `src` functions. `src` must never import Streamlit. This keeps NLP code reusable, independently testable and accessible from notebooks.

1. Students implement NLP concepts, not infrastructure. Starter code handles layouts, navigation, session state, formatting, plots, uploads, validation, environment handling, model downloads/caching/loading and deployment configuration when introduced.
2. Manual implementation comes before library comparison where educationally useful: manual TF-IDF → compare with scikit-learn.
3. One implementation in `src/` powers notebooks, tests and Streamlit. Do not duplicate algorithms in notebooks or pages.
4. Every major lab should expose Input → Intermediate Representation → Processing → Scores/Calculations → Result.
5. The same synthetic university dataset will later support TF-IDF Search → Semantic Search → RAG → Academic Assistant.
6. RAG must remain transparent: Question → Chunks → Embeddings/Retrieval → Similarity Scores → Top-K Chunks → Constructed Context → Generated Answer → Sources.
7. No paid service may be required to complete the workshop.

The shared dataset follows the canonical synthetic fact contract. Keep its facts consistent when experimenting; document generation is handled by the workshop starter materials.

Optional local generation uses the pinned Qwen2.5-0.5B-Instruct model. Retrieval remains independently usable without it. Use cached resources locally; generation can be slow and incorrect.

# Workshop Checkpoints

Overall Checkpoints0–8 are functional in reference form. Checkpoint9 (portfolio/deployment readiness) remains Task18 work. Exercise-specific milestones below do not renumber this roadmap.

### Checkpoint 0

Environment and application run successfully.

### Checkpoint 1

Text preprocessing works.

### Checkpoint 2

Text can be represented numerically.

### Checkpoint 3

TF-IDF search engine works.

### Checkpoint 4

Semantic search works.

### Checkpoint 5

Transformer experiments work.

### Checkpoint 6

Document chunk retrieval works.

### Checkpoint 7

RAG pipeline works.

### Checkpoint 8

Academic Assistant works.

### Checkpoint 9

Repository is portfolio-ready and deployment-ready.

# Exercise 1 — Text Preprocessing

### Objective

Understand how raw text is transformed before numerical representation. Inspect each operation, explain what it preserves or removes, and construct a stable vocabulary with meaningful occurrence counts.

### Reference state and eventual student starter

This development checkout contains complete, tested reference implementations. `STUDENT TODO BOUNDARY 1.1` through `1.7` identify the small educational cores for a later student starter checkpoint. They are not unfinished code. Do not replace the working functions with `pass` now. The dedicated starter branch/checkpoint strategy will be prepared later.

In that future starter, function signatures, docstrings, examples, hints, validation, library objects and resource handling will remain. Only the lines between `BEGIN STUDENT CORE` and `END STUDENT CORE` will become student TODOs. TODO 1.7 has two short cores: one for vocabulary and one for frequency. Complete one boundary at a time and run its tests.

### Concepts

- **Normalization:** make selected text conventions consistent. Here case and punctuation are separate choices.
- **Tokenization:** divide text into the units later representations use. Use a standard tokenizer, not a new tokenizer algorithm.
- **Stopwords:** frequent function words that a task may choose to remove. Filtering can also remove meaningful negation; it is not automatically required.
- **Stemming:** apply rules to reduce word forms; a stem may not be a dictionary word.
- **Lemmatization:** seek dictionary forms using a lexical resource and grammatical assumptions. A noun default may leave verbs unchanged.
- **Vocabulary:** unique tokens of the chosen representation, sorted to make later column positions deterministic.
- **Word frequency:** count every occurrence of each token. Repetition affects counts but not the unique vocabulary.

Stemming and lemmatization are **parallel alternatives on the same tokens**, not a stem-then-lemmatize sequence. Choose which representation supplies vocabulary and frequencies. POS tagging is an optional demonstration, not a student implementation task.

### Files

- Edit educational cores in `src/preprocessing.py` in the eventual starter.
- Observe results in the supplied `pages/01_Preprocessing_Lab.py` UI.
- Experiment with shared functions in `notebooks/01_preprocessing.ipynb`.
- Validate behaviour with `tests/test_preprocessing.py`.
- Leave resource setup in `src/nltk_resources.py` to the starter infrastructure.

### TODOs

#### STUDENT TODO 1.1 — Lowercase Normalization — ★ Guided

- **Objective/concept:** normalize case without performing other cleaning.
- **Location:** `lowercase_text(text)`; boundary 1.1.
- **Input:** a text string, for example `NLP is Useful!`.
- **Expected output:** `nlp is useful!`; punctuation and whitespace remain.
- **Steps:** identify the supplied string; apply only case conversion; return a string; check an empty string too.
- **Hint:** Python strings have a built-in lowercase operation. Do not tokenize here.

#### STUDENT TODO 1.2 — Punctuation Handling — ★ Guided

- **Objective/concept:** handle punctuation visibly while preserving word boundaries.
- **Location:** `remove_punctuation(text)`; boundary 1.2.
- **Input:** `human-language!`.
- **Expected output:** `human language `, including the replacement spaces.
- **Steps:** inspect each character; use the supplied punctuation predicate; replace punctuation with spaces; preserve other characters; assemble the result.
- **Hint:** deleting a hyphen entirely could merge two words. Do not lowercase or silently collapse whitespace in this function. This simple policy also splits apostrophes and decimal points.

#### STUDENT TODO 1.3 — Word Tokenization — ★ Guided

- **Objective/concept:** use a standard library tokenizer to produce ordered units.
- **Location:** `tokenize_words(text)`; boundary 1.3.
- **Input:** `natural language processing is useful`.
- **Expected output:** `["natural", "language", "processing", "is", "useful"]`.
- **Steps:** pass the text to the provided tokenizer; return its token list; retain repeated tokens and ordering.
- **Hint:** `wordpunct_tokenize` needs no downloaded Punkt model. Preserved punctuation can become tokens; tokenization does not also lowercase or remove stopwords.

#### STUDENT TODO 1.4 — Stopword Removal — ★ Guided

- **Objective/concept:** filter membership in a provided stopword set.
- **Location:** `remove_stopwords(tokens, stopword_set=None)`; boundary 1.4. The starter loads the set.
- **Input:** `["natural", "language", "processing", "is", "useful"]`.
- **Expected output with English stopwords:** `["natural", "language", "processing", "useful"]`.
- **Steps:** inspect each token; compare its lowercase form with the set; retain non-members in a new list; keep original spelling and order for retained tokens.
- **Hint:** do not remove items from the list while iterating. Passing an empty custom set should retain every token. English filtering removes `not`; discuss whether that is appropriate.

#### STUDENT TODO 1.5 — Stemming — ★ Guided

- **Objective/concept:** apply a standard stemmer to each token.
- **Location:** `stem_words(tokens)`; boundary 1.5. A PorterStemmer is supplied.
- **Input:** `["studies", "processing"]`.
- **Expected output:** `["studi", "process"]`.
- **Steps:** apply the stemmer to each token; collect outputs in the same order; retain repetitions.
- **Hint:** call the object's stemming method; do not implement Porter rules. Non-dictionary results are expected. Porter also lowercases by default.

#### STUDENT TODO 1.6 — Lemmatization — ★ Guided

- **Objective/concept:** apply the supplied lexical lemmatizer with an explicit POS assumption.
- **Location:** `lemmatize_words(tokens, pos="n")`; boundary 1.6. Resource checks and error handling are supplied.
- **Input:** `["students", "learning"]` with noun POS.
- **Expected output:** `["student", "learning"]`. With verb POS, `learning` can become `learn`.
- **Steps:** apply the lemmatizer to each original token; pass the supplied POS; keep order and repetitions; accept that some words stay unchanged.
- **Hint:** noun is the default assumption, not automatic grammatical analysis. Compare one assumption at a time; do not lemmatize the stemmed list.

#### STUDENT TODO 1.7 — Vocabulary and Word Frequency — ★★ Core

- **Objective/concept:** distinguish unique types from repeated occurrences.
- **Locations:** `build_vocabulary(tokens)` and `calculate_word_frequencies(tokens)`; the two boundary-1.7 cores.
- **Input:** `["nlp", "language", "nlp"]`.
- **Expected outputs:** vocabulary `["language", "nlp"]`; frequency mapping `{"language": 1, "nlp": 2}`.
- **Vocabulary steps:** collect distinct tokens; sort them; return a list. For several documents, flatten their token lists before calling the function.
- **Frequency steps:** start an empty mapping; visit every token; initialize an unseen token's count; increment occurrences; return the mapping with sorted keys.
- **Hint:** the sum of counts equals the input token count. Begin with the small manual counting loop; compare with `collections.Counter` afterward. Do not deduplicate before counting.

### Intended student starter forms

These signatures stay unchanged; the actual starter retains all surrounding explanations and infrastructure:

| Function | Educational core removed later | Infrastructure retained |
|---|---|---|
| `lowercase_text(text: str) -> str` | 1.1 case conversion | Text validation |
| `remove_punctuation(text: str) -> str` | 1.2 character replacement | Validation and punctuation predicate |
| `tokenize_words(text: str) -> list[str]` | 1.3 tokenizer call | Validation and imported tokenizer |
| `remove_stopwords(tokens, stopword_set=None) -> list[str]` | 1.4 membership filter | Validation, set loading and set normalization |
| `stem_words(tokens) -> list[str]` | 1.5 stemmer application | Validation and stemmer object |
| `lemmatize_words(tokens, pos="n") -> list[str]` | 1.6 lemmatizer application | Validation, WordNet check, lemmatizer and error handling |
| `build_vocabulary(tokens) -> list[str]` | 1.7 unique sorted tokens | Token validation |
| `calculate_word_frequencies(tokens) -> dict[str, int]` | 1.7 counting loop | Token validation |

Illustrative starter shape only; this is not the current source:

```python
def lowercase_text(text: str) -> str:
    """Lowercase only; preserve punctuation, digits and whitespace."""
    _validate_text(text)  # Starter infrastructure remains.
    # STUDENT TODO 1.1 — Lowercase Normalization
    # Difficulty: ★ Guided
    # Goal: Return the text with lowercase letters only.
    # Hint: Use the string's case-conversion operation.
    raise NotImplementedError("Complete STUDENT TODO 1.1")
```

The same replacement pattern applies only to the marked core in each row above, including the return inside the retained lemmatization error-handling block. No duplicate solution files are created.

### Step-by-Step Instructions

1. Complete resource setup once using the checklist.
2. Open the lab and keep the original text visible.
3. Locate each boundary and its input/output contract in the source.
4. In the eventual starter, implement one core and run the tests before continuing.
5. Compare each intermediate stage. Switch stemming/lemmatization representations without chaining them.
6. Run the notebook using the same imported functions; experiment rather than copying an algorithm into a notebook cell.
7. Run the complete test suite and commit a working checkpoint.

### Run

From the activated environment at the repository root:

```sh
python -m src.nltk_resources
python -m pytest tests/test_preprocessing.py
python -m streamlit run app.py
```

Choose **Preprocessing Lab** in the sidebar. Run `python -m pytest` for all workshop checks. Setup failures should be resolved with `python -m src.nltk_resources --download`; a missing optional POS resource does not block core processing.

### Expected Output

The lab displays raw text, lowercase text, punctuation-handled text, tokens, optional stopword filtering, stems, lemmas, sorted vocabulary and frequency counts separately. The comparison table uses the same original tokens for both morphology operations. Bypassed or unavailable stages are clearly labelled. Frequency counts reflect the representation selected in the lab.

### Experiment

- Compare `NLP nlp NLP` before and after lowercasing.
- Add repeated punctuation and a hyphen; inspect replacement spaces and token boundaries.
- Repeat a word; observe stable vocabulary size but changing frequency.
- Compare `studies`, `processing` and `learning` under stemming, noun lemmatization and verb lemmatization.
- Toggle stopword removal for `not useful`; inspect the meaning lost.
- Turn off punctuation handling; observe that punctuation can become a counted token.
- Select an input from `data/examples/preprocessing_examples.json` and explain every visible change.

### Checkpoint

**Checkpoint 1 — Text preprocessing works.** Each operation is individually testable, the vocabulary is deterministic, counts conserve token occurrences, and the lab/notebook reuse the source implementation.

### Suggested Git Commit

In the eventual student checkpoint, stage your completed source change, then commit:

```sh
git add src/preprocessing.py
git commit -m "Complete text preprocessing lab"
```

# Exercise Template

Future tasks will populate this template with concrete instructions.

## Exercise X — Name

### Objective

To be supplied with the exercise.

### Concept

To be supplied with the exercise.

### Files You Will Edit

Relevant `src/` files will be identified.

### TODOs

Assigned IDs and difficulty levels will be listed.

### Step-by-Step Instructions

To be supplied with the exercise.

### Run

Exercise-specific test and lab commands will be supplied.

### Expected Output

To be supplied with the exercise.

### Experiment

To be supplied with the exercise.

### Checkpoint

The associated milestone will be identified.

### Suggested Git Commit

Use a short description of the completed concept, once assigned.

# Documentation Synchronization Contract

Every future implementation task introducing a student TODO must update all five artifacts together:

1. Source code
2. `workshop/TODO_INDEX.md`
3. `docs/STUDENT_GUIDE.md`
4. `docs/INSTRUCTOR_GUIDE.md`
5. Relevant tests

This is a permanent project rule. Register only TODOs that actually exist; record their IDs, difficulty, source location, objective and validation instructions.


# Exercise 2 — One-Hot Encoding, Bag of Words and N-Grams

## Objective and connection

Exercise 1 produced tokens. Now build numerical coordinates for words and documents:
Tokens → Vocabulary → Vocabulary Index → One-Hot → BoW → Document-Term Matrix → N-Grams → CountVectorizer verification.

Edit `src/classical_nlp.py`; inspect the provided `pages/02_BoW_NGrams_Lab.py` and `notebooks/02_bow_ngrams.ipynb`. The repository contains working reference implementations. A later starter checkpoint removes only the small BEGIN/END student cores, preserving signatures, hints, validation and UI. All four TODOs are ★★ Core, Day 1.

Inputs are token lists, not raw text. A corpus is a list of token lists. Reuse Exercise 1 before calling representation functions. Unlike Exercise 1's flat-token `build_vocabulary`, Exercise 2's function accepts multiple tokenized documents. Import from the intended module.

## TODO 2.1 — Multi-document vocabulary

- Concept/goal: create one sorted coordinate system shared by the corpus.
- Location: `src/classical_nlp.py`, `build_vocabulary(documents)`.
- Input: `[["nlp", "processes", "language"], ["nlp", "analyzes", "text"]]`.
- Expected: `["analyzes", "language", "nlp", "processes", "text"]`.
- Steps: collect each document's tokens into one sequence; reuse Exercise 1's unique vocabulary utility. Do not collect characters from raw strings.
- Hint: flatten exactly one level; retain repeated tokens until uniqueness is applied.
- Experiment: reverse document order and repeat a token; vocabulary must remain unchanged.

## TODO 2.2 — One-hot encoding

- Concept/goal: encode one vocabulary term with one active coordinate.
- Location: `src/classical_nlp.py`, `one_hot_encode(word, vocabulary)`.
- Input: word `nlp`, vocabulary `["language", "nlp", "text"]`.
- Expected: `[0, 1, 0]`.
- Steps: allocate a vocabulary-sized zero vector, find the word's index using the provided mapping, set that position to one and return the vector.
- Hint: vector length is vocabulary size; sum is exactly one for a known word.
- Experiment: reorder the vocabulary; predict which coordinate moves.

## TODO 2.3 — Manual Bag of Words

- Concept/goal: represent a complete document by occurrence counts.
- Location: `src/classical_nlp.py`, `bag_of_words(tokens, vocabulary)`.
- Input: `["nlp", "language", "nlp"]` with `["language", "nlp", "text"]`.
- Expected: `[1, 2, 0]`.
- Steps: allocate zeros; visit every token; increment its mapped position; return counts. Use the shared vocabulary, not a new vocabulary per document.
- Hint: repetitions must accumulate; an absent vocabulary term stays zero.
- Experiment: repeat `nlp` once more and predict the changed coordinate. Shuffle tokens and compare results.

## TODO 2.4 — N-Grams

- Concept/goal: preserve limited local order through contiguous windows.
- Location: `src/classical_nlp.py`, `generate_ngrams(tokens, n)`.
- Input: `["natural", "language", "processing"]`, `n=2`.
- Expected: `[("natural", "language"), ("language", "processing")]`.
- Steps: determine all starts that permit a complete window; slice n tokens at each start; store each window as a tuple in order.
- Hint: the final valid start is token count minus n. Do not cross documents.
- Experiment: try n=1, n=3 and n=4; predict one-element tuples, a full trigram, then an empty result.

## Provided utilities and edge policies

`create_vocabulary_index` exposes word → coordinate; `build_document_term_matrix` stacks your BoW vectors. `countvectorizer_matrix` provides the library comparison after manual work. No separate TODO is assigned for those utilities.

Both one-hot and BoW raise `ValueError` for unknown words. Duplicate vocabulary terms are rejected. Supply unique vocabulary in any explicit order; the functions preserve that order. Empty documents produce zero rows, empty corpus vocabulary is empty, and empty vocabulary can represent only empty documents. n must be a positive integer; oversized n produces no windows.

## Run and observe

1. Complete cores 2.1–2.4 in the eventual student starter.
2. Run `python -m pytest tests/test_classical_nlp.py`.
3. Run `python -m streamlit run app.py` and open BoW NGrams Lab.
4. Inspect the default BOW-01 corpus, vocabulary index, one-hot table, count vectors and matrix before reading the library comparison.
5. Run the notebook from the repository or notebooks directory.

For BOW-01 the vocabulary has six terms and the matrix has three rows. Manual and library counts should match. The lab retains stopwords and does not stem/lemmatize, to isolate representation. CountVectorizer is configured with the same token lists and coordinate mapping. Its raw-text defaults can lowercase and discard single-character tokens; a callable analyzer bypasses those defaults and ngram_range. Equivalent comparisons require equivalent inputs and aligned columns.

## Experiment and mini challenge

Compare Dog bites man with Man bites dog: identical unigram counts, different bigrams. Compare car and automobile: separate coordinates do not convey their related meaning. Add repeated common words and observe how counts dominate. Increasing N-Gram range creates more possible dimensions.

Predict before running the notebook's challenge: D1 Students learn natural language processing; D2 Students learn machine learning; D3 Machine learning analyzes data. Determine vocabulary size, students one-hot, BoW vectors, most repeated terms and D1 bigrams, then verify with your shared functions.

## Checkpoint and commit

Checkpoint 2: **Text can be represented numerically.**

Suggested commit: `git commit -m "Complete one-hot BoW and N-Grams lab"`.

We can now represent documents numerically, but every occurrence contributes equally. Common words may dominate the representation. How can we measure which words are actually informative?


# Exercise 3 — TF-IDF From Scratch

## Objective and files

Are all words equally informative? Build on Exercise 2's shared vocabulary and counts. Edit only the educational cores in `src/classical_nlp.py`; inspect `pages/03_TFIDF_Search_Lab.py` and `notebooks/03_tfidf.ipynb`. The current checkout contains complete references; a later starter conversion preserves signatures, validation, assembly and UI. All IDs below are Day 1, ★★ Core except 3.5 (★★★ Challenge).

## Manual convention and paper-and-pencil example

TF(t,d) = count(t,d) / number of document tokens.
DF(t) = number of documents containing t, counted once per document.
IDF(t) = ln(N / DF(t)), natural logarithm, with no smoothing or added constant.
TF-IDF(t,d) = TF(t,d) × IDF(t).

Calculate before coding: D1=`common nlp nlp`, D2=`common text`, D3=`common text`. Sorted vocabulary is `[common,nlp,text]`. For nlp in D1: count=2, length=3, TF=2/3, DF=1, N=3, IDF=ln(3)≈1.098612, TF-IDF≈0.732408. common has DF=3, hence IDF=0. text has DF=2, IDF≈0.405465, and D2 weight≈0.202733. Write the three matrix rows on paper, then verify in the explorer.

## TODO 3.1 — Term Frequency

- Objective/function: `calculate_tf(tokens, vocabulary)` normalizes existing BoW counts.
- Input: `[nlp,studies,language,nlp]`, columns `[language,nlp,studies]`.
- Expected: `[0.25,0.5,0.25]`.
- Steps: reuse the provided counts; divide each by total document tokens; keep column order. Empty document returns zeros.
- Hint: use document length, not vocabulary size. Repeating nlp changes both its numerator and the denominator.

## TODO 3.2 — Document Frequency

- Objective/function: `calculate_df(documents, vocabulary)` counts document presence.
- Input: `[[nlp,nlp,language],[nlp,text],[language,processing]]` with columns `[language,nlp,processing,text]`.
- Expected: `[2,2,1,1]`.
- Steps: inspect the provided count matrix; for each column count rows whose count is positive.
- Hint: ten occurrences in one document still contribute only one to DF. Add repeats and verify DF stays unchanged.

## TODO 3.3 — Inverse Document Frequency

- Objective/function: `calculate_idf(document_frequencies, document_count)` applies ln(N/DF).
- Input: DF `[3,2,1]`, N=3. Expected: `[0,0.405465,1.098612]` approximately.
- Steps: compute N divided by each DF, take natural logarithm, retain coordinate order.
- Hint: a corpus-wide term gets zero, rare terms get larger weights. Starter validation rejects zero/negative DF, DF>N and invalid N. Corpus vocabulary should contain only observed terms; do not invent an IDF for an absent term.

## TODO 3.4 — TF-IDF

- Objective/function: `calculate_tfidf(tokens, vocabulary, idf)` combines local frequency and corpus rarity.
- Input: D1 above, columns `[common,nlp,text]`, IDF `[0,ln(3),ln(1.5)]`.
- Expected: `[0,0.732408,0]` approximately.
- Steps: reuse the provided TF result; multiply each coordinate by the matching corpus IDF. Starter `build_tfidf_matrix` calls this function for each row.
- Hint: do not recompute counts or IDF independently. Reorder both vocabulary and IDF together to inspect alignment.

## TODO 3.5 — Query TF-IDF Vector

- Objective/function: `build_query_tfidf_vector(tokens, vocabulary, idf)` reuses the corpus space.
- Input: query `[nlp,robotics]`, same example vocabulary/IDF.
- Expected: `[0,1.098612,0]`; robotics creates no coordinate.
- Steps: retain only known tokens, then reuse TF-IDF with the existing vocabulary and IDF. Query TF denominator is retained known-token count. Empty or completely OOV query gives a vocabulary-sized zero vector.
- Hint: never build a query vocabulary or fit IDF on the query. Repeat a known word and try entirely novel words. Search tolerates unknown terms deliberately; Exercise 2 one-hot/BoW remain strict so coordinate mistakes are visible.

## Run and experiment

Run `python -m pytest tests/test_classical_nlp.py`, then `python -m streamlit run app.py`. Open TFIDF Search Lab, inspect Learning Corpus before Academic Course Corpus. Read BoW, TF, DF, IDF and TF-IDF separately. Use the selected document/term explorer to trace count → length → TF → DF → N → IDF → TF-IDF. Try a corpus-wide term and a rare term.

The library comparison comes afterward. sklearn uses raw counts × [ln((1+N)/(1+DF))+1], `smooth_idf=True`, `sublinear_tf=False`, `norm=None` here. Default sklearn uses L2 normalization; we disable it for readable weights. Even `smooth_idf=False` retains +1, so it is not our classroom formula. Identical tokens/columns do not imply equal numbers. Both demonstrate rarity weighting. Do not treat differences as implementation errors.

# Exercise 4 — Build Your First Search Engine

## Objective and files

Compare a query with document vectors and return deterministic top-K results. TODO 4.1 is in `src/classical_nlp.py`; TODOs 4.2 and 4.3 are in `src/retrieval.py`. All are ★★ Core, Day 1. Starter orchestration provides preprocessing, corpus preparation, dictionary results, metadata and UI; no class hierarchy or LLM.

## TODO 4.1 — Cosine Similarity

- Objective/function: `cosine_similarity(vector_a, vector_b)` measures vector direction.
- Formula: dot(A,B) / (magnitude(A) × magnitude(B)); magnitude(A)=sqrt(sum of squared coordinates).
- Input: A=[1,1], B=[1,0]. Expected: 1/sqrt(2)≈0.707107.
- Steps: multiply matching coordinates and sum; calculate each magnitude; form denominator; return zero if denominator is zero, otherwise divide.
- Hint: identical/proportional nonzero vectors give one, orthogonal vectors give zero. Scores are similarities, not probabilities.

## TODO 4.2 — Query-Document Similarities

- Objective/function: `score_documents(query_vector, document_vectors)` compares every row.
- Input: query=[1,0], rows=[[1,0],[0,1],[0,0]]. Expected: [1,0,0].
- Steps: iterate rows in corpus order and call the shared manual cosine function for each. Do not duplicate cosine math.
- Hint: score positions must continue to refer to the original document positions; predict the first result before sorting.

## TODO 4.3 — Rank Documents

- Objective/function: `rank_documents(documents, scores, top_k)` orders results descending.
- Input: D1,D2,D3 with scores [0.5,0.8,0.8], K=2. Expected: D2 then D3.
- Steps: sort positions by score, descending; preserve original order on ties; take top K and copy metadata with score.
- Hint: stable sorting preserves ties. Do not mutate corpus order. K larger than the corpus returns available results, K=0 gives no results; negative K is invalid.

## Predict, run and inspect

Learning Corpus D1 covers natural language processing, D2 machine learning, D3 computer networks, D4 database systems. Predict D1 for `natural language processing`, then inspect query tokens, ignored OOV terms, shared query vector, dot product, magnitudes, every score and ranked text. Matched terms explain lexical evidence. Top-K includes zero-score rows transparently; these are not positive matches. Empty/OOV queries and queries containing only corpus-wide terms give zero vectors and deterministic ties.

Run `python -m pytest tests/test_retrieval.py`; use the shared notebook to repeat the calculation and switch to the 20 course descriptions. Course search indexes descriptions only; titles label results and keywords are not added. Try the suggested exact queries and the human-communication paraphrase. Do not assume a course-title query succeeds when its description uses other wording. Retrieval returns ranked documents; it does not produce summaries, answer cross-document questions or analyze paper frequencies.

## Experiment and library ranking

Compare a query with itself and with a scaled version. Try `nlp robotics`, then a completely novel query. Compare a rare term and a corpus-wide term. Use sklearn's corpus IDF for its query, rather than fitting on the query. Normalized query TF differs from raw counts only by a positive scale and therefore leaves cosine unchanged; different IDF may change the direction and ranking. Read the instructor baseline after predicting outcomes.

# Checkpoint 3 — TF-IDF Search Engine Works

> You have now built a small information retrieval system using NLP and mathematics—without an LLM.

Suggested commit: `git commit -m "Build TF-IDF search engine"`.

TF-IDF still depends on shared terms. Humans connect human language with human communication; lexical vectors may not. Word embeddings are a later exercise.


# Exercise 5 — Word Embeddings and Word2Vec

## Objective and connection

Sparse one-hot dimensions do not express semantic closeness; TF-IDF depends on vocabulary overlap. Task 06's human-communication paraphrase ranked Distributed Systems above NLP. Word2Vec changes the representation, learning dense word coordinates from repeated contexts. Cosine stays the same comparison operation. This does not yet solve sentence/document retrieval.

The distributional hypothesis says words occurring in similar contexts tend to develop related representations. Context windows select nearby words. CBOW predicts a target from context (natural ___ processing → language); Skip-Gram predicts nearby context from a target (language → natural, processing). Gensim handles the neural training algorithm and optimization. Students implement the workflow, not backpropagation.

## Files and settings

Edit `src/embeddings.py`; inspect the provided Word Embeddings Lab and `notebooks/04_word_embeddings.ipynb`. Training data is `data/examples/word2vec_corpus.json`: sentence-like spans derived from existing synthetic course names/descriptions and academic prose, with per-span source paths. It is supplied, not a student data-writing exercise. No ground-truth labels/benchmark expected answers are training input.

Default: vector_size=50, window=5, min_count=1, sg=0, epochs=100, seed=42. Starter fixes workers=1, a stable initialization hash, sample=0.001, negative=5, hs=0, alpha=0.025, min_alpha=0.0001, sorted_vocab=1, batch_words=10000, shrink_windows=True. Change one parameter at a time; exact cross-platform floating-point outputs are not promised. The current checkout retains complete bounded references; the eventual starter removes only student cores and preserves validation, caching, explanations and PCA.

## TODO 5.1 — Prepare Tokenized Sentences (★ Guided)

- Goal/function/file: `prepare_tokenized_sentences(sentences)` in `src/embeddings.py` prepares Gensim input using Exercise 1.
- Input: `["Students study NLP!", "Machine learning analyzes data."]`.
- Expected: `[["students","study","nlp"],["machine","learning","analyzes","data"]]`.
- Steps: lowercase each sentence, replace punctuation with spaces, call the existing word tokenizer, discard empty token lists. Retain stopwords and unstemmed word forms for this experiment.
- Hint: Gensim receives a list of token lists, not one raw string or flat token list.
- Experiment: add capitalization, punctuation-only text and an empty sentence; inspect structure without duplicating preprocessing.

## TODO 5.2 — Train a Tiny Word2Vec Model (★★ Core)

- Goal/function/file: configure `train_word2vec(tokenized_sentences, ...)` in `src/embeddings.py` using Gensim.
- Input: tokenized supplied corpus plus small settings.
- Expected: a locally trained model with vocabulary and vectors of configured size; no guaranteed semantic ranking.
- Steps: pass tokenized sentences, vector size, context window, min_count, sg, epochs and seed to the Word2Vec constructor, retaining provided stable-hash/training settings.
- Hint: sg=0 is CBOW; sg=1 is Skip-Gram. More epochs means more training passes; min_count filters rare terms. The starter prevents empty/one-word retained vocabularies.
- Experiment: compare architectures while holding all other settings fixed. Do not implement the optimizer or caching.

## TODO 5.3 — Retrieve a Word Vector (★ Guided)

- Goal/function/file: `get_word_vector(model, word)` in `src/embeddings.py` retrieves a known term's dense values.
- Input: trained model and lowercase language.
- Expected: copied list of vector_size floats, not a vocabulary-sized one-hot list.
- Steps: access the model's keyed vectors for the selected word and convert to a list compatible with manual cosine.
- Hint: keyed vectors are exposed through model.wv; starter validation handles OOV.
- Experiment: compare one-hot length and active position with dense size and first ten coordinates. Individual coordinates have no simple semantic labels.

## TODO 5.4 — Find Similar Words (★★ Core)

- Goal/function/file: `find_similar_words(model, word, top_n)` in `src/embeddings.py` returns learned nearest neighbours.
- Input: model, language, top_n=5.
- Expected: up to five `(word, similarity)` pairs, excluding language itself; exact neighbours may vary.
- Steps: call the keyed-vector nearest-word operation using the validated word and capped count supplied by infrastructure.
- Hint: nearest is contextual association, not guaranteed synonymy or factual connection.
- Experiment: compare language, attendance and database; inspect rare-word counts when a neighbour seems surprising.

## TODO 5.5 — Calculate Word Similarity (★★ Core)

- Goal/function/file: `calculate_word_similarity(model, word_a, word_b)` in `src/embeddings.py` connects embeddings to Task 06 cosine.
- Input: model, language and text; provided vector retrieval produces two equal-dimensional lists.
- Expected: manual cosine approximately equal to Gensim similarity; self similarity approximately one.
- Steps: reuse the existing cosine function over the two supplied vectors. Compare with model.wv.similarity in the lab/notebook afterward.
- Hint: representation changed, not the similarity formula; do not reimplement cosine.
- Experiment: language/text, machine/learning, machine/database, student/course and attendance/examination. Record measured outcomes rather than predict a guaranteed winner.

## Run, inspect and experiment

Install updated requirements, run `python -m pytest tests/test_embeddings.py`, then `python -m streamlit run app.py`. Open Word Embeddings Lab. Inspect corpus/provenance, training settings, vocabulary, dense and one-hot vectors, manual versus Gensim scores, neighbours, architecture comparison and optional PCA. Caching is provided outside src; identical settings do not retrain for word selections. After editing the training core, use the provided **Clear cached models and retrain** control to rebuild models. No pretrained embeddings are downloaded.

| Experiment | Setting A | Setting B | Observation |
|---|---|---|---|
| Architecture | CBOW | Skip-Gram | Write your measured changes |
| Context window | 2 | 5 | Write your measured changes |
| Vector size | 20 | 50 | Write your measured changes |

Small windows may emphasize local relationships; larger ones may capture broader topics. Larger vectors have capacity but need data and can be noisy. These are intuitions, not guarantees. Keep the same corpus/seed/epochs when comparing.

## Interpretation and limitations

This model uses a small synthetic corpus; similarities demonstrate learning, not production-quality judgments. OOV words raise ValueError with guidance. Ordinary Word2Vec cannot generate vectors for arbitrary unseen words. PCA is starter code: **2D projection for visualization only**; reducing dimensions can distort closeness. Select 2–20 known words, preferably 8–20, and inspect original-space similarities too.

GloVe learns from global co-occurrence statistics. FastText uses character subword pieces (learn, learning, learner), helping morphology and rare/unseen forms. These are conceptual comparisons only; no downloads or TODOs.

bank approved the loan; students sat on the river bank. Word2Vec gives bank one static vector if known, rather than choosing a sense from sentence context. One word → one vector. The notebook verifies identical stored vectors for a known word in two source contexts; bank itself need not exist in this corpus. Do not confuse word vectors with complete sentence representations.

## Exercise 5 milestone — Words Have Learned Vector Representations

This is the Exercise 5 word-representation milestone. Semantic search was pending at Task 07; Exercise 7 now satisfies the original workshop semantic-search Checkpoint 4. The two milestone names describe distinct outcomes.

Suggested commit: `git commit -m "Explore Word2Vec embeddings"`.

How do we represent an entire sentence as one meaningful vector? Sentence embeddings are next; do not implement them in this exercise.

# Exercise 6 — Sentence Embeddings and Semantic Similarity

## Objective and concepts

Word2Vec represents individual words. Our questions and passages need complete-text representations. A pretrained sentence encoder composes contextual representations into one fixed-size dense vector per sentence/passage. Our workshop corpus is too small to train a strong general-purpose encoder, so we use free local inference with `sentence-transformers/all-MiniLM-L6-v2`. Transformer architecture is taught later. Sentence vectors are not averages of the workshop Word2Vec vectors.

The comparison is still the manual cosine from Exercise 4: dot product divided by the product of vector magnitudes. Only the representation changed. Similarity is not a probability, a truth check, or proof that two claims mean exactly the same thing.

## Before you begin

Install updated requirements. While online, preload with `python -m src.sentence_resources --download`; then verify with `python -m src.sentence_resources`. Normal loading reads only the local `.cache/sentence_transformers/` directory. No account, token, paid API or GPU is required. See the setup checklist for offline copies. Do not put model setup inside your TODOs.

## Files and boundaries

Edit `src/embeddings.py`, observe `pages/05_Sentence_Embeddings_Lab.py`, and experiment in `notebooks/05_sentence_embeddings.ipynb`. The checkout remains complete working/reference state. A later starter checkpoint will remove only the bounded cores, preserving validation, loading, caching and signatures. After source edits, use the lab's reload button to clear model/vector caches.

### TODO 6.1 — Encode Sentences (★★ Core)

- **Goal/function/location:** `encode_sentences` in `src/embeddings.py`; turn complete texts into a batch of vectors using a provided model.
- **Input:** a list such as `["Students study NLP.", "Computers process language."]`. A single sentence must still be a one-item list.
- **Expected output:** numeric array with shape `(2, 384)`; one text gives `(1, 384)`. Empty list gives `(0, 384)` without inference. Blank strings are rejected.
- **Steps:** keep original texts; send the full list to the model's batch encoder; request NumPy output; keep progress display off; do not request extra normalization. Inspect rows, columns and finite values.
- **Hint:** use the supplied encoder's encoding method. Do not iterate over the characters of a raw string. Do not lowercase, remove stopwords, stem, or create a query vocabulary first.
- **Experiment:** encode one sentence, then five. Which axis changes? What happens when sentence length changes?

The model itself includes a Normalize module. `normalize_embeddings=False` means no additional normalization; it does not remove that module. Measured output norms are therefore about 1. We preserve the exact pretrained encoder and still calculate cosine explicitly; no separate raw pooling representation is substituted.

### TODO 6.2 — Calculate Sentence Similarity (★★ Core)

- **Goal/function/location:** `sentence_similarity` in `src/embeddings.py`; compose encoding and the existing manual cosine.
- **Input:** two non-empty sentence strings and the supplied model.
- **Expected output:** one finite cosine value. Identical texts should score approximately 1, not necessarily bit-for-bit 1 on every machine.
- **Steps:** encode both texts together; identify row A and row B; convert rows to the list format accepted by the existing manual cosine; calculate and return the score.
- **Hint:** one row is a complete text, not a token. Reuse the existing function instead of a library similarity shortcut.
- **Experiment:** rewrite a sentence without changing its meaning, then compare it with an unrelated subject. Predict before observing.

## Run

```sh
python -m streamlit run app.py
python -m pytest tests/test_sentence_embeddings.py
```

Open Sentence Embeddings Lab. Run the notebook cells in order using the same environment. Model integration tests never download: they skip with guidance if cache is absent. Before class, prepare the model and require these tests to pass with **no skips**.

## Experiments and interpretation

Use the four preset pairs and fill this table. Custom text is also supported.

| Pair | Sentence A | Sentence B | TF-IDF | Sentence embedding | Observation |
|---|---|---|---:|---:|---|
| Similar wording | Students study natural language processing. | Students learn natural language processing. | ? | ? | ? |
| Paraphrase | Natural language processing analyzes human language. | Computers learn to work with human communication. | ? | ? | ? |
| Reversed roles | The student teaches the instructor. | The instructor teaches the student. | ? | ? | ? |
| Unrelated | Natural language processing analyzes text. | Computer networks route packets between devices. | ? | ? | ? |

The pairwise TF-IDF baseline uses classroom normalized TF × ln(N/DF), fitted on the eight preset texts plus distinct custom inputs. A two-text-only fit gives every shared term IDF zero; this larger inspectable reference collection avoids that artifact. It is a different fitted IDF from Task 06's 20-course corpus. The failure-case section reuses Task 06's actual corpus IDF and only compares the query with NLP and Distributed Systems descriptions. It does not rank every course semantically.

Find a paraphrase, an unrelated pair, and a high-overlap pair with different meaning. High semantic similarity can still miss negation or who did what. Text longer than 256 model word pieces is truncated. Do not interpret arbitrary dimensions as named properties.

## Checkpoint and commit

**Exercise 6 milestone — Semantic Sentence Similarity Works.** Overall Checkpoint5 remains Transformer experiments; exercise milestones do not renumber the roadmap.

```sh
git commit -m "Add sentence embedding similarity"
```

Next question: can we compare one query with every course description? Semantic Search is the next exercise, not implemented here.

# Exercise 7 — Semantic Search

## Objective and core idea

Exercise 6 compared one complete text with another. Semantic search compares one query with many document vectors. Each of twenty existing course descriptions becomes one row of a (20,384) matrix. The same exact pretrained encoder maps the query into the same space. Cosine gives one score per course; stable descending ranking selects Top-K. The model, revision, native normalization and CPU/offline policy are unchanged.

```text
TF-IDF Search: Query → lexical vector → cosine → rank
Semantic Search: Query → semantic vector → cosine → rank
```

What stayed the same? Discover it by tracing both pipelines: cosine, ranking and Top-K. What changed? Representation. This is retrieval, not answering or generation.

## Files and setup

Edit `src/retrieval.py`; observe `pages/06_Semantic_Search_Lab.py` and execute `notebooks/06_semantic_search.ipynb`. Reuse Exercise 6 encoding and Exercise 4 cosine/ranking. Both systems use description text only from `data/search_corpus/course_descriptions.csv`; labels such as title/code/semester are not secretly added to one method. No new packages or model downloads are required beyond Task 08 preflight. If the cache is missing, run `python -m src.sentence_resources --download` explicitly while online and verify locally without the flag.

The repository remains complete working/reference state. Later starter preparation removes only the four small cores; signatures, validation, cache, UI and model setup remain. Use the retrieval lab's cache-clear button after editing corpus/encoding cores. Caching belongs to starter infrastructure, not your TODOs.

### TODO 7.1 — Encode Corpus Documents (★★ Core)

- **Goal/function/location:** `encode_corpus_documents` in `src/retrieval.py`, preserving CSV row order.
- **Inputs:** preloaded model and document dictionaries with ID, text and optional metadata.
- **Expected output:** N×384 numeric matrix, one row per text; empty corpus gives (0,384). Blank document text is rejected.
- **Steps:** collect text fields in original order; call the existing batch sentence encoder; keep metadata outside the encoded strings.
- **Hint:** do not prepend titles or keywords. Do not retrain Word2Vec or load another model.
- **Experiment:** swap two document positions and re-encode; confirm their embedding rows move with them.

### TODO 7.2 — Encode Search Query (★ Guided)

- **Goal/function/location:** `encode_search_query` in `src/retrieval.py`; use the identical encoder for query and documents.
- **Input:** one non-empty query string, retaining original wording/case/punctuation.
- **Expected output:** a list of 384 floats. Blank/whitespace-only queries raise clear guidance; the UI checks before searching.
- **Steps:** wrap the query in a one-item batch; reuse the existing encoder; take its first row in the list format accepted by manual cosine.
- **Hint:** a (1,384) batch and a (384,) vector are different shapes.
- **Experiment:** paraphrase a query and inspect which coordinates/ranks change.

### TODO 7.3 — Calculate Semantic Similarities (★★ Core)

- **Goal/function/location:** `calculate_semantic_similarities` in `src/retrieval.py`; compare the query with every row.
- **Inputs:** 384-dimensional query list and N×384 document matrix in document order.
- **Expected output:** N finite cosine scores in the same order, not sorted yet.
- **Steps:** adapt array rows to the existing list representation; reuse Exercise 4 scoring, which calls manual cosine for each row; retain the original row order.
- **Hint:** a single pairwise score is insufficient; one query needs N comparisons. Reuse zero-vector safety.
- **Experiment:** query with an actual document text; inspect its self similarity near 1.

### TODO 7.4 — Retrieve Top-K Semantic Results (★★ Core)

- **Goal/function/location:** `retrieve_semantic_results` in `src/retrieval.py`; reuse existing deterministic ranking.
- **Inputs:** original document metadata, aligned scores and K (default 5).
- **Expected output:** dictionaries containing preserved metadata and a similarity score. Higher first; equal scores retain original order. K=0 gives []; oversized K gives all documents; negative/non-integer K is invalid.
- **Steps:** pass aligned metadata and scores through the shared descending ranking function; retain the first K entries.
- **Hint:** semantic representation does not require a new ranking algorithm or vector database.
- **Experiment:** change K without changing the query; verify the shared prefix and stable ties.

## Run and trace

```sh
python -m streamlit run app.py
python -m pytest tests/test_semantic_retrieval.py
```

Inspect query text → query vector preview → (20,384) corpus matrix → all twenty cosine scores → sorted indices → Top-K. Compare both ranked lists for the same query and K. Scores are similarity, not confidence or probability. Comparing absolute score magnitudes across systems does not establish which result is better.

## Experiments

| Experiment | Query | TF-IDF Top-1 | Semantic Top-1 | More appropriate? Why? |
|---|---|---|---|---|
| Exact keywords | natural language processing | ? | ? | ? |
| Paraphrase | Which subject teaches machines to work with human communication? | ? | ? | ? |
| Ambiguous | learning | ? | ? | ? |
| Unrelated | How do I bake a chocolate cake? | ? | ? | ? |
| Custom | Your own query | ? | ? | ? |

Try machine learning, database systems and computer communication networks too. Semantic search need not win. Nearest neighbours exist even for unrelated text; this exercise does not implement relevance thresholds or automatically abstain. TF-IDF all-zero ties are also not evidence of an answer. Semantic results have no invented token-level explanation.

The RET-03–RET-06 presets reuse benchmark wording, but this app corpus has no attendance policy, calendar or question papers. Instructor evaluation uses the separate original eighteen full Markdown files to evaluate their expected source IDs. Benchmarks and measured results are evaluation artifacts, never application answers. The notebook demonstrates this distinction; no analyzer, summarizer or cross-document answer is implemented.

## Evaluation concepts

For a clearly specified single source, Top-1 asks whether it ranks first; Top-3 hit asks whether it appears in the first three. MRR averages 1/rank of the first relevant result. Multi-source evidence coverage must also track how many required sources were retrieved; one hit does not mean a multi-source question is answered. Summary/comparison/paper queries only evaluate candidate source retrieval, not those unimplemented tasks.

## Checkpoint and commit

**Overall Checkpoint4 — Semantic Search Works.** Document chunk retrieval later satisfies overall Checkpoint6.

Text → Numbers → TF-IDF Search → Word Embeddings → Sentence Embeddings → Semantic Search.

```sh
git commit -m "Build semantic search"
```

Next: Day 2 introduces Transformers and then meaningful retrieval units for full academic documents. Chunking, RAG and generation are not implemented in this exercise.
# Exercise 8 — Understanding Attention

## Objective and connection

Yesterday, changing vector representations changed retrieval behavior. Today we ask how representations incorporate context. Word2Vec gives `bank` one vector, although a river bank and a financial bank differ. Attention selects and combines surrounding representations.

**Simplified educational attention example:** our invented vectors are not learned BERT/GPT attention; attention weights are not explanations of model reasoning.

## Concepts and files

Edit `src/transformers_nlp.py`; observe the provided Transformer Lab and `notebooks/07_transformers.ipynb`. The repository contains completed reference cores; a later starter checkpoint removes only their bounded lines. Validation, UI, composition, plotting and model setup remain starter code. Do not replace working functions with `pass` now.

Query asks what information is needed; keys advertise matching information; values contribute information. These are vector roles, not literal questions, database keys or dictionary values.

## TODO 8.1 — Calculate Attention Scores (★★ Core)

- Function/location: `calculate_attention_scores`, boundary 8.1 in `src/transformers_nlp.py`.
- Goal/formula: compute `score_i = Q · K_i` for every key.
- Input: Q `[1,0]`; keys `[[1,0],[0,1],[0.5,0.5]]`.
- Expected output: scores `[1,0,0.5]`, one per key in the same order.
- Steps: multiply matching coordinates, sum each key's coordinate products, collect the scores. Dimension checks are provided.
- Hint: recall dot products from cosine similarity; here do not divide by norms.
- Experiment: change Q to `[0,1]`. Predict which key scores highest.

## TODO 8.2 — Softmax Attention Weights (★★ Core)

- Function/location: `softmax_attention_weights`, boundary 8.2 in the same module.
- Goal/formula: `w_i = exp(s_i) / sum_j exp(s_j)`.
- Input: the complete score vector `[1,0,0.5]`.
- Expected output: about `[0.506480,0.186324,0.307196]`, summing to one.
- Steps: subtract the largest score from every score; exponentiate shifted values; divide by their combined sum.
- Hint: subtraction preserves relative ratios and prevents overflow. Normalize the entire vector, not each scalar independently.
- Experiment: add 10000 to every score; weights should be unchanged. Compare equal scores.
- Precision note: mathematical weights are positive; extreme gaps can round tiny computed weights to zero. Empty scores give an empty vector.

## TODO 8.3 — Weighted Context Representation (★★ Core)

- Function/location: `calculate_weighted_context`, boundary 8.3 in the same module.
- Goal/formula: `context = sum_i w_i × V_i`.
- Input: the weights above; values `[[2,0],[0,2],[1,1]]`.
- Expected output: about `[1.320157,0.679843]`, with the dimension of a value vector.
- Steps: start an all-zero context; multiply each value row by its corresponding weight; add the contributions. Row-count and non-negative unit-sum checks are provided.
- Hint: one weight scales every coordinate of its value row.
- Experiment: change only V1. Scores and weights stay unchanged, but context changes.

## Paper-and-pencil calculation

Compute Q·K1 = 1, Q·K2 = 0 and Q·K3 = 0.5. The exponential denominator is approximately `2.718282 + 1 + 1.648721 = 5.367003`. Calculate the three weights and add weighted value rows. Predict the context before checking the lab. A zero query produces uniform weights, not automatically a zero context.

## From this example to Transformers

Self-attention derives Q/K/V from tokens in the same sequence using learned projections. An attention-matrix row is a token attending across other tokens. Real scaled dot-product attention uses `softmax(QKᵀ / sqrt(d_k))V`; scaling limits large dot products as dimensions grow. The lab checkbox compares scaling on the toy example; no new TODO is required. Multiple heads use different learned projections and can learn different relationships, without guaranteed grammatical labels.

## Run and experiment

Run `python -m pytest tests/test_transformers_nlp.py` and `python -m streamlit run app.py`. Open Transformer Lab. Change Q, keys and values separately; record changes in scores, weights and context. Manual attention needs no model download.

# Exercise 9 — Explore Pretrained Transformers

## Objective and architecture

There is **no algorithm TODO** here. Experiment with supplied CPU models and interpret output critically. Tokenization → token embeddings and positional information → stacked blocks with self-attention, feed-forward layers, residual connections and normalization → task output.

BERT means Bidirectional Encoder Representations from Transformers. It is typically encoder-oriented and learns contextual representations, including through masked language modeling: `The student submitted the [MASK] before the deadline.` GPT means Generative Pre-trained Transformer. It is typically decoder-oriented, uses causal context and predicts successive next tokens. GPT is conceptual only; no generative model is installed.

| Feature | BERT | GPT |
|---|---|---|
| Typical architecture | Encoder | Decoder |
| Context style | Bidirectional | Causal/autoregressive |
| Pretraining intuition | Masked tokens | Next token |
| Common strength | Understanding/representation | Generation |
| Example uses | Classification, NER, extractive QA | Generation, assistants |

These are typical distinctions, not absolute restrictions. MiniLM from the sentence/semantic-search labs already used Transformer encoders and sentence-similarity training. Sentence vectors and token task outputs are different products of related technology.

## Setup and files

`src/transformer_resources.py` provides pinned models, explicit preload, local-only loading and process caching. Do not implement resource handling. Before class run `python -m src.transformer_resources --download`, then verify offline with `python -m src.transformer_resources`. Read the setup checklist. About 796 MB additional model disk is required; no paid credentials or GPU. Missing task models display guidance while attention and previous labs remain available.

## Experiments

Select a task and click **Run selected task**. These are genuine local inference outputs.

| Task | Inputs to try | Observe |
|---|---|---|
| Sentiment | Positive, negative, mixed or sarcastic feedback | Label, task score, ambiguity and domain mismatch |
| NER | Person, institution, location, course/technical terms | Category, original text, span, misses and false positives |
| Extractive QA | Change question/context; ask an unsupported question | Answer slice, score, positions, unsupported spans |

QA reads only short context supplied by you. It does not retrieve a document or generate prose. The SQuAD-v1 model can return an incorrect span when a question is unsupported; its score is not factual confidence. Check every answer against context. Changing supplied context is an experiment, not an edit to canonical university facts.

Use at most 400 tokens per input, including combined question/context for QA; questions are limited to 64 tokens. Excessive text is rejected rather than silently truncated. No extraction, chunking or retrieval-to-QA integration is introduced.

| Experiment | Prediction | Measured label/span and score | Correct? Why? |
|---|---|---|---|
| Positive/negative/mixed sentiment | ? | ? | ? |
| Person vs technical term in NER | ? | ? | ? |
| Supported vs unsupported QA | ? | ? | ? |

## Checkpoint and transition

**Overall Checkpoint5 — Transformer experiments work.** RAG later satisfies overall Checkpoint7.

Suggested commit: `git commit -m "Explore attention and transformers"`.

Retrieval and task inference remain separate capabilities. What if we do not know which document supplies context? Before combining evidence with a language model, we need meaningful document units. Document processing and chunking come next; do not implement them here.


# Exercise 10 — Document Extraction, Cleaning and Chunking

**IMPLEMENTED / READY (working reference).** This prepares documents; it does not retrieve chunks or generate answers.

## Objective and concept

Documents → Extracted Text → Conservative Cleaning → Word Windows → Overlap → Source Metadata. Long documents can exceed the shared encoder's window. Small passages preserve inspectable evidence, but splitting alone does not guarantee semantic completeness.

## Files you will edit and observe

Edit `src/document_processor.py`. Observe `pages/08_RAG_Lab.py` and `notebooks/08_rag.ipynb`. Markdown in `data/academic_documents/` is authoritative. Three PDFs in `data/sample_pdfs/` are derived teaching copies, not additional independent university facts.

## TODOs

Only bounded educational cores are removed in the eventual student starter. The current repository keeps complete working reference implementations. Keep function signatures, validation, offset helpers and UI unchanged.

| TODO | Function | Difficulty | Goal and input → output |
|---|---|---|---|
| 10.1 | clean_document_text | ★ Guided | Extracted string → readable whitespace-normalized string |
| 10.2 | split_text_into_chunks | ★★ Core | Text and positive word size → ordered windows, including final partial text |
| 10.3 | create_overlapping_chunks | ★★ Core | Text, size and overlap → windows retaining shared boundary words |
| 10.4 | attach_chunk_metadata | ★★ Core | Windows and source identity → labeled passages with provenance |

### TODO 10.1 — Clean extracted text

Normalize carriage-return/newline conventions first. Collapse horizontal whitespace within each line, trim line ends, and retain at most one blank line. Preserve case, punctuation, numbers and headings: `Attendance:\t75%.` becomes `Attendance: 75%.`. Hint: treat horizontal spaces separately from newlines. Experiment with multiple blank lines and a policy sentence. Do not reuse aggressive classical stopword/stemming transformations on evidence.

### TODO 10.2 — Split text into chunks

The supplied offset helper preserves the exact text substring for a word window. Generate starts at zero, size, twice size, and so on; let the helper shorten the last window. `a b c d e f g`, size 3, should produce `a b c`, `d e f`, `g`. Empty text returns an empty list. Hint: iteration stops before the word count, not before the last full window. Experiment with exact multiples and very short documents.

### TODO 10.3 — Add overlap

Advance by `step = chunk_size - overlap`. Append a window, then stop if it reaches the final word; otherwise advance. For `a b c d e f g h`, size 4, overlap 1: `a b c d`, `d e f g`, `g h`. The shared words are `d` and `g`. Do not produce a redundant extra tail made only of already-covered overlap. Starter validation enforces integer `0 <= overlap < size`. Zero overlap reuses TODO 10.2. Experiment with 0/20/40 overlap at size 100.

### TODO 10.4 — Attach chunk metadata

Copy each window, then add document ID/title, running chunk index, source type/path and real page number. Combine document identity with a padded index for the chunk ID. Hint: numbering continues across PDF pages, while word offsets restart within each page. Do not mutate the original windows. Experiment by tracing a chunk back to its exact cleaned source substring.

## Provenance and counts

Word/character starts are zero-based; ends are exclusive. Offsets refer to cleaned text, not raw PDF bytes. Whitespace-delimited words include Markdown symbols and differ from model tokens. Markdown has no physical pages (`page_number=None`). PDF pages are numbered from 1; empty pages keep their place, and overlap never crosses a physical page. IDs such as `HCE-DOC04::chunk_000` are deterministic for unchanged source/settings but positional: re-chunking can change their content.

## Run

```text
python -m pytest tests/test_document_processor.py
python -m streamlit run app.py
```

PDF parsing/upload handling, guards, tokenizer setup, statistics and UI are starter infrastructure. Text-based unencrypted PDFs up to 10 MiB / 50 pages are supported; image-only scans need OCR, which is outside this exercise. Uploads remain in memory. Optional token inspection uses the existing cached MiniLM tokenizer without embeddings/downloads. If missing, use the earlier explicit model preload; extraction/chunking still work.

## Expected output and experiment

Default **100 words / 20-word overlap** yields 152 chunks across the 14 canonical documents. MiniLM counts include special tokens: no default chunk exceeds 256 on this dataset. This is measured corpus behavior, not a promise for arbitrary text.

Compare sizes 60/100/240 and overlaps 0/20/40; record chunk count, boundary context, duplicated positions, minimum/average/maximum length and token outliers. Inspect an attendance policy chunk, then a PDF page transition. Does a sentence get cut? Does overlap preserve enough context? Count duplicated positions separately from repeated terms. Do not use benchmark answers as retrieved content.

## Checkpoint and suggested Git commit

**Exercise 10 milestone — Documents Are Ready for Retrieval.** This prepares overall Checkpoint6 without renumbering the roadmap.

```text
git commit -m "Prepare documents for chunk retrieval"
```

Next we will embed and retrieve chunks; no Task 12 student TODO is introduced here.

Exercise 10 boundary demonstration: the canonical sentence “Students must maintain at least 75% attendance in each enrolled course.” split at seven words gives “Students must maintain at least 75% attendance” / “in each enrolled course.” With overlap three, the second window is “least 75% attendance in each enrolled course.” Compare the threshold and per-course qualification; overlap improves continuity without making every window a complete rule. The lab derives this sentence from the canonical attendance document.

# Exercise 11 — Semantic Chunk Retrieval

**IMPLEMENTED / READY (working reference).** Retrieval returns evidence, not a generated answer. Context construction and generation remain planned.

## Objective and connection

Task 09 encoded a course description as one vector. This exercise encodes each passage as one vector: 14 canonical Markdown documents → 100-word chunks / overlap 20 → 152 passages → 152 × 384 matrix. The query and passages use the same pinned MiniLM encoder. The mathematics is still cosine; the unit of retrieval changed from document to chunk.

```text
Semantic Search: query → embedding → compare with document embeddings → rank
RAG Retrieval: question → embedding → compare with chunk embeddings → rank evidence
```

## Files and TODOs

Edit only the bounded cores in `src/retrieval.py`. Model loading, token checks, cache identity, UI and validation are supplied. Inspect `pages/08_RAG_Lab.py` and the shared `notebooks/08_rag.ipynb`. Keep the repository in working/reference form until the later student-starter checkpoint.

| TODO | Function | Difficulty | Input → output |
|---|---|---|---|
| 11.1 | encode_document_chunks | ★★ Core | Ordered chunk records → N × 384 matrix |
| 11.2 | encode_user_question | ★ Guided | Non-empty question → 384-dimensional vector |
| 11.3 | calculate_chunk_similarities | ★★ Core | Question vector + chunk matrix → N cosine scores |
| 11.4 | retrieve_top_k_chunks | ★★ Core | Chunks, scores, K → copied evidence with score/rank |

### TODO 11.1 — Encode document chunks

Extract text fields from each chunk in original order. Pass that ordered list to the shared sentence encoder. Return the matrix without prepending IDs, titles or expected answers. Empty input gives (0,384); blank chunk text is rejected. Hint: row i must correspond to chunk i. Experiment: encode one chosen chunk separately and compare its row.

### TODO 11.2 — Encode the user question

Reuse the existing query-encoding helper with the same model object used for chunks. It returns one vector rather than a one-row batch and rejects blank questions. Hint: a different model would change the coordinate space. Experiment: paraphrase an attendance question without changing its meaning.

### TODO 11.3 — Calculate question-chunk similarities

Compare the question vector with each row in order, using the existing manual cosine path. Return one finite score per row. Zero vectors score zero. Hint: scoring and row alignment are unchanged from semantic course search. Experiment: compare scores for related and unrelated passages. Similarity is not probability or confidence.

### TODO 11.4 — Retrieve Top-K evidence

Reuse descending stable ranking, then attach 1-based ranks to copied results. Preserve every original chunk field, including document/chunk identity, title, source type/path, physical page where available, offsets and text. Ties keep original chunk order. K=0 returns []; K larger than the corpus returns all; negative/non-integer K fails. Hint: document IDs repeat across chunks, but chunk IDs must remain unique. Experiment: compare K=1/3/5/8. Do not deduplicate neighboring chunks yet.

## Run and inspect

```text
python -m pytest tests/test_chunk_retrieval.py
python -m streamlit run app.py
```

Select RAG Lab. The default retrieval corpus is all 14 canonical Markdown sources with unchanged 100/20 settings; the earlier selected-document processing controls are a separate experiment. For PDF retrieval, choose the selected-document/PDF corpus. Do not mix derived PDF duplicates into the canonical corpus.

Before encoding, tokenizer checks verify fit within 256 model tokens. Oversized settings fail clearly; they do not silently truncate. Cached MiniLM is required for retrieval; processing remains usable without it. Changing question or K reuses the corpus matrix. Changed source text, settings or model identity/revision invalidate the relevant cache. No new download or dependency is needed.

Follow question → vector preview/norm → chunk matrix shape → all scores → sorted indices → Top-K text → provenance. Inspect the actual text rather than judging results by score alone. Markdown has no physical page number; PDF page provenance remains truthful.

## Experiments

Try a direct attendance fact, an NLP prerequisite paraphrase, the broad NLP-completion question, a cake recipe and your own question. Fill in:

| Question | Top-1 Source | Top-3 Sources | Best Score | Useful Evidence? |
|---|---|---|---:|---|
| Direct fact | | | | |
| Paraphrase | | | | |
| Multi-source | | | | |
| Unrelated | | | | |
| Custom | | | | |

Find two overlapping neighbors in the results. More evidence is not automatically more useful evidence. Top-K finds the nearest passages even when the corpus has no answer. There is no universal threshold or automatic abstention.

## Checkpoint and commit

**Overall Checkpoint6 — Relevant document chunks can be retrieved.** Checkpoint9 still means portfolio/deployment readiness.

```text
git commit -m "Add semantic chunk retrieval"
```

Next: combine retrieved evidence into controlled context. No Task 13 TODO is introduced here.

# Exercise 12 — Construct Grounded RAG Context

**READY — complete working/reference form.** Exactly three small student cores are identified in `src/rag.py`; infrastructure and the optional model adapter remain provided.

## Objective and concepts

Retrieval returns ordered evidence dictionaries. Format them as labelled context, then combine that evidence with rules and the question into a prompt. **Context** is evidence. **Prompt** is instructions + context + question. Neither is the generated answer.

| TODO | Function | Difficulty | Input → output |
|---|---|---|---|
| 12.1 | build_context | ★★ Core | Ordered, already-budgeted chunks → readable context |
| 12.2 | format_source_block | ★★ Core | One chunk and number → labelled provenance/text block |
| 12.3 | build_grounded_prompt | ★★★ Challenge | Question + context → inspectable grounded prompt |

### TODO 12.1 — Combine chunks

Keep the supplied order. Format each chunk with the source-block helper, number from 1, and join blocks with visible boundaries. Do not sort again, summarize, merge or deduplicate. Hint: gather blocks first, then join them. Expected: one string; empty input gives ''. Try reversing inputs deliberately and observe the order.

### TODO 12.2 — Preserve source labels

A block begins `[SOURCE n]`, then Document, Document ID, Chunk ID and Page fields, a blank line and unchanged evidence text. Markdown page is N/A; a PDF uses its physical page number. Hint: title/IDs come from metadata, not from guessed text. Preserve all evidence numbers and wording. Scores, paths and offsets remain in a separate mapping; similarity is omitted from the LLM evidence because it is not factual authority.

### TODO 12.3 — Build the grounded prompt

Combine provided grounding instructions, explicitly delimited context, the original question and an ANSWER section. Rules require context-only facts, insufficient-context behavior and relevant source labels; document text is evidence, not instructions overriding the rules. Hint: use the provided rules rather than inventing unsupported facts. Empty evidence has an explicit marker. Inspect the entire prompt. These instructions do not guarantee correctness or solve prompt injection.

## Files and run

Edit `src/rag.py`. Observe RAG Lab and `notebooks/08_rag.ipynb`; both reuse the shared implementation.

```text
python -m pytest tests/test_rag.py
python -m streamlit run app.py
```

## Context budget and experiments

Starter policy defaults to **5 chunks / 8000 context characters**, counting labels, metadata and separators. Include only whole blocks in rank order; stop at the first block that does not fit. Exclusions and reasons are visible. No text is cut mid-block, and no duplicate evidence is silently removed. Characters/words, MiniLM tokens and generation-model tokens are different quantities.

Try attendance at K=1/3/5. Read included sources, duplicate document blocks, context size and exclusions. Test a smaller character budget. Try the cake query and NLP-credit query at the unchanged K=5. Prompt design cannot recover genuinely missing evidence. The credit query misses the expected syllabus, but retrieved lab guidelines supply the corresponding four-credit theory relationship. A source miss need not be a fact miss. Compare retrieved evidence with generated output separately.

## Checkpoint

**Exercise checkpoint 10 — Retrieved Evidence Becomes a Grounded Prompt.** This is not the final RAG checkpoint. Overall RAG/deployment milestones remain pending.

```text
git commit -m "Build grounded RAG context"
```

# Exercise 13 — Connect a Local Language Model

**Instructor-guided optional integration — no low-level algorithm TODO.** The interface is simply `generate(prompt) → text`; loading, CPU selection, chat templating, caching and token generation are starter code in `src/local_llm.py` / `src/llm_resources.py`.

The optional backend uses the pinned public Qwen2.5-0.5B-Instruct checkpoint with existing Transformers/PyTorch. It adds about 1.00 GB of cache and substantial CPU RAM. Preload explicitly only if the classroom machine has room:

```text
python -m src.llm_resources --download
python -m src.llm_resources
```

After preload, inference is local/offline and needs no account, paid API or GPU. Opening the app never downloads or loads this model. The generation button loads it when clicked; missing/broken resources do not remove retrieval, context or prompt inspection.

Notebook/test demonstrations may use `RecordingTestLLM`, clearly labelled TEST ONLY; its fixed response is simulated, not NLP inference or an application fallback. A small real model can hallucinate, ignore rules or omit/misuse source labels. An emitted label is not a verified citation. Inspect actual evidence. Task 14 will assemble and examine the full transparent RAG experience.

# Exercise 14 — Inspect a Complete RAG Pipeline

**READY — integration, experimentation and interpretation. No new algorithm TODOs.**

## Objective and definition

Retrieval-Augmented Generation retrieves external information and supplies it as context to a generative model before an answer is produced. Our implementation combines document preparation/retrieval (Tasks 10–12), context and prompt augmentation (Task 13), and optional local generation (Task 13). Task 14 makes every stage inspectable.

Question → Question Embedding → Top-K Evidence → Included Context → Grounded Prompt → Optional Generated Answer → Sources Supplied to the Model.

## Files and run

Reuse completed algorithms in `src/`; do not duplicate them. Integration helpers live in `src/rag.py`; the complete lab is `pages/08_RAG_Lab.py` and the final progression is in `notebooks/08_rag.ipynb`.

```text
python -m streamlit run app.py
python -m pytest tests/test_rag.py
Open the RAG Lab and ask the instructor for recorded evaluation results.
```

The evaluator defaults to retrieval/context/prompt only. Add `--generate` only after the earlier explicit optional Qwen preload. It never downloads models. No Qwen is required for core tests, retrieval, context or prompt work.

## Primary workflow

1. Choose/edit a question and K; defaults remain 5 with 100-word / 20-word-overlap chunks.
2. Choose **Retrieval + Prompt Only** or **Full Local Generation**. Run the pipeline; the latter generates only on request when the optional model is installed.
3. Inspect **Generated Answer** separately from source evidence.
4. Read **Sources Supplied to the Model**: pipeline-generated SOURCE labels, document/chunk/page/rank and expandable original text.
5. Expand retrieved evidence, embedding/similarity details, included/excluded context, exact prompt, model/timings and diagnostics.
6. Change K/budget/source and rerun. Previous answers are invalidated when relevant inputs change, so they cannot be shown beside new evidence.

Sources are deterministic provenance, not verified citations. The model-label inspector detects known labels, unknown labels or no labels. A label that exists in the supplied context does not prove the associated claim is supported. Diagnostics describe counts, source diversity, overlaps, exclusions, generation availability and label syntax, not correctness.

## Failure diagnosis

First ask whether the knowledge base contains an answer; an **unsupported question** is a separate case. If an answer exists: is useful evidence in Top-K? If not, **retrieval failure**. Was that evidence included in the actual context? If not, **context/budget failure**. Is the generated answer supported? If not, **generation failure**. Are labels absent or unknown? Record **attribution failure** separately; deterministic supplied sources still remain visible. More than one failure can occur.

RAG does not guarantee correct/complete retrieval, reasoning, attribution, freedom from hallucinations or currency unless the knowledge base is updated. Similarity is not confidence. Prompt rules and valid source labels do not certify facts.

## Experiments

| Question | K | Useful Evidence Retrieved? | Answer Correct? | Labels Emitted? | Failure Type |
|---|---:|---|---|---|---|
| Attendance | 1 | | | | |
| Attendance | 5 | | | | |
| Cake | 5 | | | | |
| Attendance/examination multi-source | 5 | | | | |
| Your own question | 5 | | | | |

Also compare attendance K=3 and a one-chunk context budget at K=5. Read all blocks for NLP credits: a missed preferred syllabus may coexist with indirect lab-guideline support. Inspect RET-06's network evidence and whether the generated answer actually names the right subject. Do not assume a good source rank guarantees generation quality.

## Checkpoint

**Overall Checkpoint7 — Complete Transparent RAG Pipeline.** Generation is optional. Assistant integration satisfies overall Checkpoint8; portfolio/deployment readiness remains Task18.

```text
git commit -m "Complete transparent RAG pipeline"
```

# Exercise 15 — Question Paper Analyzer

> Use the simplest method that reliably solves the problem.

This structured-paper task uses deterministic phrase rules, not embeddings, Transformers or an LLM. Extraction → normalization → canonical taxonomy matching → one primary topic → frequency → ranking → filtering. Implementation stays in working/reference form.

## Objective and files

Understand a small explainable classifier and inspect its limitations. Edit only the bounded cores in `src/question_analyzer.py`; parsing, validation, reporting and Streamlit are supplied. Observe `pages/09_Question_Paper_Analyzer.py`. No ninth notebook is needed; teach this exercise through the app and guides.

## TODO map — preserve 14.x numbering

Exercise number **15** intentionally uses TODOs **14.1–14.5**.

| TODO | Function | Difficulty | Input → output |
|---|---|---|---|
| 14.1 | normalize_question_text | ★ Guided | Original question → consistent matchable text |
| 14.2 | match_question_topics | ★★ Core | Text + canonical taxonomy → explicit matches, primary and reason |
| 14.3 | count_topic_frequencies | ★★ Core | Analyzed records → primary-only counts |
| 14.4 | rank_topics | ★ Guided | Observed counts → descending deterministic topic ranking |
| 14.5 | filter_questions_by_topic | ★★ Core | Topic selection → copied question records with provenance |

### TODO 14.1 — Normalize

Casefold and normalize Unicode. Turn punctuation/hyphens into phrase boundaries, separate joined TFIDF, and collapse whitespace. Preserve technical letters/digits: Word2Vec, CBOW, BERT and GPT stay recognizable. No stemming or stopword removal. Example: `Explain Multi-Head Self-Attention.` → `explain multi head self attention`. Hint: normalize separators rather than deleting them and joining adjacent words.

### TODO 14.2 — Match topics and aliases

Use supplied canonical names/aliases with word boundaries. Collect every explicit match and preserve its phrase/type/position. Narrow CBOW/Skip-Gram/BERT distinctions have specificity tier 2; Word2Vec/Attention tier 1; other topics tier 0. Choose greater specificity, then longer phrase in words, then canonical taxonomy order; retain equal-specificity/length candidates as ambiguous. Hint: related concepts are not aliases and must not propagate labels. The matcher supports a limited final-word singular/plural surface variant, so Transformer/Transformers match. No paper or question ID may influence selection.

A multi-head self-attention question can match the canonical self-attention alias and Transformer, yet select Attention. Why is that reasonable? Inspect the matched phrase instead of assuming every word is a semantic match. No lexical match returns **UNMATCHED**, not a forced topic.

### TODO 14.3 — Count

Initialize canonical-topic counts and an unmatched counter. Increment only the primary topic once per top-level question. Frequency = number of questions assigned that primary. Do not count every secondary match or read target frequencies. Hint: related-topic mentions do not create extra slots.

### TODO 14.4 — Rank

Sort frequency descending; taxonomy order breaks ties. Zero-count canonical topics remain visible. UNMATCHED is a separately reported status, not a seventeenth topic. Hint: keep ties reproducible rather than relying on unordered sets.

### TODO 14.5 — Filter

Default to primary-topic equality and retain paper/question IDs, original text and explanation. Optional secondary mode includes explicitly matched topics only. Hint: return full copied records, not bare strings. Try Attention, Transformers, Word2Vec and TF-IDF, then compare primary-only versus secondary filtering.

## Run and inspect

```text
python -m pytest tests/test_question_analyzer.py
python -m streamlit run app.py
```

Open Question Paper Analyzer. Four actual Markdown papers yield 12 top-level questions each, 48 total, with 60 marks per paper. `## Qn — 5 marks` headings define slots; instructions/subparts are not new questions. Examine original/normalized text, primary, all aliases, ambiguity/reason and provenance. Then inspect counts, ranking, per-paper matrix and filters.

Canonical taxonomy source is `data/university_facts.json#/nlp_topic_taxonomy`: 16 topics. GPT belongs to Language Models; RAG is not a separate category. Production projects names/aliases only. Ground-truth metadata is evaluation-only and never supplies classifications. The optional UI evaluation section is explicitly labelled Observed Analyzer Result versus Known Synthetic Ground Truth.

## Experiments and limits

| Question | Matched Phrase | Primary Topic | Do You Agree? |
|---|---|---|---|
| A paper question | | | |
| Your paraphrase | | | |
| A question naming two topics | | | |

| Topic | Observed Count | Rank |
|---|---:|---:|
| Transformers | | |
| Attention | | |
| TF-IDF | | |

Try `CBOW`, `Skip-Gram`, `Word2Vec`, `word embeddings`, `GPT`, an unrelated question and a paraphrase with no canonical alias. The baseline correctly matches 39/48 questions, leaves nine unmatched and reports those gaps honestly. A component-only IDF question may remain unmatched because IDF is not a canonical alias by itself.

**Question frequency in these synthetic papers is a learning aid and does not predict future examination questions.** Four undated synthetic papers do not establish real historical trends.

## Checkpoint and commit

**Exercise checkpoint 12 — Question Paper Analyzer Works.** This is a feature milestone, not final Academic Assistant or deployment readiness.

```text
git commit -m "Build question paper analyzer"
```

Task 16 will integrate the existing academic capabilities; no integration work is part of this task.

# Final Integration Exercise — Intelligent Academic Assistant

No new algorithm TODOs: the 49 existing IDs are frozen. Use page 10 to integrate four capabilities:

| Request | Method | Result |
|---|---|---|
| Academic rules and facts | Existing transparent RAG | Evidence, optional generated answer, supplied sources |
| Course / curriculum navigation | Same academic document retrieval | Semester, prerequisite and course evidence |
| Previous-paper questions | Existing deterministic analyzer | Actual extracted questions and observed counts |
| Study / revision support | Retrieved syllabus + optional Qwen | Evidence or clearly labelled generated study content |

Choose Auto, Academic Q&A, Previous Papers or Study Support. Auto prioritizes historical-paper requests, then strong summary/revision requests, then academic/navigation terms, then weak study intent. A prerequisite request containing “study before” remains academic navigation. Unknown intent falls back to academic retrieval with an explicit explanation. Explicit modes override Auto. Inspect the displayed routing reason and matched rules in Diagnostics.

Try attendance, NLP semester, prerequisites and examination date. Read the supplied evidence before trusting optional generation. Sources identify context supplied to the model; they are not verified citations. Another course's individual exam date may be unspecified—do not infer one from an examination window.

Try “Show Transformer questions”, “How many Word2Vec questions appeared?”, “most frequent topics”, and “Compare topics across previous papers”. These reuse primary-topic analysis, not metadata answers or Qwen. Transformers currently yields eight historical questions. Nine questions remain unmatched; exact aliases have limitations. Counts describe four synthetic papers and do not predict future exams. GPT uses the existing Language Models topic. Unknown topics are reported without forcing a match.

Try “Summarize NLP Unit 4” and “Create five revision questions for NLP Unit 4”. Enable local generation only if preloaded. New questions are labelled **AI-Generated Revision Questions**, never previous-paper records. Without Qwen, inspect retrieved syllabus evidence instead. Without MiniLM, previous-paper analysis still works. Without paper data, academic retrieval remains independent.

## Integration experiments

1. Compare Auto with explicit modes; explain a deliberately ambiguous revision-from-previous-papers request.
2. Change the question/mode after a result: the old result clears until you ask again.
3. Compare a historical Transformer filter with generated Unit 4 revision questions. Identify provenance.
4. Disable generation and inspect evidence, context and prepared sources.
5. Inspect a weakly relevant result or an unsupported request. Similarity is not factual confidence.
6. Run `python -m streamlit run app.py`; inspect routing and supplied-evidence observations, not an invented answer-accuracy score.

The complete assistant now reuses your NLP components. Suggested commit: `git commit -m "Integrate intelligent academic assistant"`. Tasks 17–18 will handle broader reliability and release presentation.

Before trusting a Unit 4 summary or generated revision questions, check that the source text actually covers Unit 4. The measured integration run retrieved other units for that short request. Generated output can therefore be fluent and still be wrong. Use this as an evidence-inspection experiment, not a model answer to memorize.


# Task17 — Reliable Classroom Use

Use `python -m workshop.preflight` for a fast offline inventory. Use `python -m workshop.preflight --full-classroom` to require MiniLM and all three Day2 task-model caches; Qwen always stays optional. Add `--smoke` for local inference checks without Qwen. Missing caches are not installed automatically. See the setup checklist for explicit online preload commands.

The Academic Assistant now checks explicit unit requests before optional generation. It derives the course scope from supplied syllabus titles and requires a matching Unit heading in supplied context. If absent or ambiguous, it withholds generation and shows the evidence/diagnostic. It does not rewrite your question, reorder results, fabricate unit text or change cosine. This applies to all unit numbers/courses, not just NLP Unit4. A heading is necessary but does not prove complete coverage. The RAG Lab still demonstrates pure semantic retrieval and its failures.

Short prerequisite queries can miss facts. Try a full question and full course name, then inspect all prerequisites—not just the first chunk. No query expansion was introduced. Keep model output separate from verified facts and historical paper records.

Interactive preprocessing/BoW inputs are limited to5,000 characters for readable classroom experiments; learning TF-IDF uses at most20 documents/10,000 characters. Sentence input shows model-token counts when truncation matters; use chunks for long passages. Algorithms themselves retain their existing APIs. Uploaded PDFs stay in memory and must be text-based, unencrypted, ≤10MiB and ≤50pages; OCR is outside scope.

There are49 algorithm IDs,49 unique boundary headers and50 labelled core regions: TODO1.7 has vocabulary and frequency subregions. Reference code is still complete; no starter stripping has occurred. Overall checkpoints keep their original0–9 meanings. The 2-day path is operational; deployment/portfolio packaging remains pending.

# Workshop Navigation and Completion — Task18

Day1: Words→Numbers→Search→Embeddings→Meaning. Day2: Transformers→Documents→Retrieval→RAG→Analysis→Assistant. Overall checkpoints0–9 retain their original meanings; the schedule and commit journey are in workshop/WORKSHOP_SCHEDULE.md.

| Exercise | Objective / concept | Edit / TODOs | Run tests | Expected structure / experiment | Milestone / suggested commit |
|---|---|---|---|---|---|
|1|Inspect separate preprocessing operations|src/preprocessing.py;1.1–1.7|python -m pytest tests/test_preprocessing.py|Tokens, sorted vocabulary, counts; change case/negation/POS|Checkpoint1; Add text preprocessing |
|2|Map vocabulary to coordinates/counts/windows|src/classical_nlp.py;2.1–2.4|python -m pytest tests/test_classical_nlp.py -k "vocabulary or one_hot or bow or ngrams or matrix"|Vectors/DTM; compare word order|Checkpoint2; Add bag of words representations |
|3–4|Weight terms and compare query direction|src/classical_nlp.py + src/retrieval.py;3.1–3.5 /4.1–4.3|python -m pytest tests/test_classical_nlp.py tests/test_retrieval.py|TF=count/length, DF=document presence, IDF=ln(N/DF), TF-IDF=TF×IDF, cosine=dot/(normA×normB); inspect rare/OOV terms|Checkpoint3; Build TF-IDF search |
|5|Learn dense word representations|src/embeddings.py;5.1–5.5|python -m pytest tests/test_embeddings.py|Vectors/neighbors; vary architecture/window/size|Word milestone; Add word embedding experiments |
|6|Represent complete texts|src/embeddings.py;6.1–6.2|python -m pytest tests/test_sentence_embeddings.py|N×384; compare paraphrases/negation using same cosine|Sentence milestone; Add sentence similarity |
|7|Compare lexical and semantic rankings|src/retrieval.py;7.1–7.4|python -m pytest tests/test_semantic_retrieval.py|Same corpus/query, scores/Top-K; try human communication|Checkpoint4; Build semantic search |
|8|Compute attention components|src/transformers_nlp.py;8.1–8.3|python -m pytest tests/test_transformers_nlp.py|Dot scores, stable softmax, weighted values; vary inputs|Attention milestone; Add attention and transformer NLP |
|9|Experiment with provided pretrained inference|No algorithm TODO; Transformer Lab|python -m src.transformer_resources|Labels/entities/span; compare unsupported context|Checkpoint5; Add attention and transformer NLP |
|10|Prepare retrievable document windows|src/document_processor.py;10.1–10.4|python -m pytest tests/test_document_processor.py|Clean text, offsets, overlap/provenance; compare windows|Document milestone; Process academic documents |
|11|Retrieve chunks using the shared encoder|src/retrieval.py;11.1–11.4|python -m pytest tests/test_chunk_retrieval.py|N×384, query vector, scores, stable Top-K; inspect weak evidence|Checkpoint6; Add semantic chunk retrieval |
|12|Construct inspectable context/prompt|src/rag.py;12.1–12.3|python -m pytest tests/test_rag.py|Ordered source blocks and grounded prompt; vary budgets|Context milestone; Build grounded RAG context |
|13–14|Integrate optional generation transparently|No new algorithm TODO; RAG Lab|python -m streamlit run app.py|Evidence/context/output distinction; compare missing model/K1/K5|Checkpoint7; Complete transparent RAG pipeline |
|15|Extract topics/count observed historical records|src/question_analyzer.py;14.1–14.5|python -m pytest tests/test_question_analyzer.py|Primary topics/ranking/filters; inspect unmatched wording|Analysis milestone; Build question paper analyzer |
|Final|Assemble existing capabilities|No new algorithm TODO; Academic Assistant|python -m streamlit run app.py|Visible routing/provenance/withheld generation; compare modes|Checkpoint8; Integrate academic assistant |

Detailed goals, inputs, formulas, steps and hints are in the exercise chapters above. Implement each small marked core while preserving its signature; read tests to predict structural/numerical outputs, then experiment in its notebook and page. Do not copy instructor solutions or implement infrastructure. In a generated starter, pages pause at expected NotImplementedError messages; after implementing notebook TODOs restart the kernel and run again. Unexpected coding errors should still be investigated with tests.

Publish only work you completed: use your own screenshots, record one meaningful failure, keep synthetic-data attribution and choose your repository/license deliberately. See docs/GITHUB_PORTFOLIO_GUIDE.md and docs/DEPLOYMENT_GUIDE.md. Checkpoint9 means a genuinely portfolio/deployment-ready project, not an unverified hosted claim. No remote publication or deployment has been performed by Task18.

Reference/release validation now includes574 passing tests on both Python versions. The generated starter has2 passing setup-smoke tests and340 collected correctness cases; those correctness cases are intentionally red until the required cores are implemented. Models are not bundled. The repository license is awaiting owner selection. Keep completed work and real evidence distinct from incomplete starter features or unverified deployment claims.
