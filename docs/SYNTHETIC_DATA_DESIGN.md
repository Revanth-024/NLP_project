# Synthetic Data Design

## Purpose and fictional setting

`data/university_facts.json` is the authoritative fact contract for **Hindu College of Engineering**, a completely fictional educational institution created only for this workshop. Velmora and Teralin Learning District are invented teaching locations. Faculty names and all academic details are original synthetic choices, not representations of real people or institutions.

Every future academic document and question paper must explicitly describe the institution as fictional and include this exact statement:

**Synthetic educational data created for NLP workshop purposes.**

Task 02 established structured facts. Task 03 provides 14 canonical Markdown academic documents, a 20-course CSV search corpus, three input-example files and four synthetic question papers. Task 04 adds preprocessing only; retrieval and RAG remain unimplemented. Markdown remains the primary knowledge source; PDFs are reserved for a later selective extraction exercise.

## Canonical-fact philosophy

Load the UTF-8 JSON using Python's standard `json` module. Its explicit fields are the source of truth; prose documents are derived views. Before adding a new policy, course fact or calendar date to a document, add it to the contract and run the consistency tests. Do not invent unsupported answers or copy real university wording.

Course codes, faculty IDs and topic IDs are stable identifiers. Benchmark `fact` references use RFC 6901 JSON pointers; update affected references if the schema changes. A reference identifies the canonical fact, not a future document or chunk ID. Syllabus `topic_ids` resolve to names and aliases in the taxonomy.

Some relationships are deliberately expressed in both directions for easy lookup: semester/course membership, faculty/course assignments and theory/laboratory links. Tests enforce agreement. Policy thresholds and examination marks have one authoritative definition; labs and courses reference it.

## Curriculum and faculty relationships

The compact eight-semester B.Tech CSE curriculum has 47 courses and 160 credits. Listed electives are fixed choices for the fictional cohort; alternate options are outside scope. It is a teaching curriculum, not an accreditation claim.

Machine Learning in Semester 5 builds on Probability and Statistics, Design and Analysis of Algorithms, and Linear Algebra for Computing. NLP in Semester 6 requires Artificial Intelligence and Machine Learning. Deep Learning in Semester 7 requires Machine Learning and Linear Algebra for Computing. Each has a linked laboratory in the same semester. Laboratory theory links are **corequisites**, not prerequisites; prerequisites must always be from earlier semesters.

The NLP syllabus has five units: text preparation, count-based representation, distributed word representation, sequence/contextual models, and meaning/retrieval. RAG is a later workshop application, not a formal syllabus unit. Five fictional faculty cover the important AI/NLP and selected retrieval-teaching courses. Other course assignments are unspecified; empty `faculty_ids` do not imply that courses lack instructors.

## Attendance and examination rules

The single minimum attendance is 75% per course, including labs, projects and seminars. Compare the unrounded attendance percentage. Shortage makes a student ineligible for that course's end-semester evaluation. There is no condonation or separate lab threshold. Verified recording errors may be corrected; approved absence does not automatically become attendance. A shortage requires re-enrollment at the next offering.

Every course uses the same 100-mark scheme: 40 internal and 60 end-semester marks. Passing requires both at least 50 total marks and at least 24 end-semester marks, plus attendance eligibility. No separate internal minimum applies. Theory uses written assessment; labs use practical demonstration/viva; projects and seminars use review/portfolio/demo. Continuous work affects internal marks rather than creating another examination-eligibility rule. Absolute grade bands apply only after the pass conditions are met.

Placement preparation remains accessible with pending courses; recruitment registration has separate final-year, CGPA and pending-course requirements. There are no employer claims or placement guarantees.

## Calendar assumptions

The academic year is 2026–2027. Odd semesters (1, 3, 5, 7) run concurrently for different cohorts, followed by even semesters (2, 4, 6, 8). This is not eight sequential semesters within a year. Teaching, internal exams, practical exams, end-semester exams and breaks have ordered, inclusive ISO date ranges.

The NLP end-semester examination has a specific date in the even-term examination window. Other courses have term windows only; do not invent individual dates. Reassessment dates and the following academic year's commencement are intentionally unspecified.

## Dataset reuse and retrieval benchmarks

The same future documents will support TF-IDF Search → Semantic Search → RAG → Academic Assistant → Question Paper Analyzer. Reusing the setting helps students compare representations while the facts stay constant.

`data/document_manifest.json` inventories the 14 academic documents and four papers with stable IDs, data-relative filenames, categories, fact areas and intended uses. It stores no full content. `data/retrieval_benchmarks.json` preserves the fifteen canonical queries and maps them to these actual document IDs. Three supplementary queries cover ML versus Deep Learning, NLP Unit 3 summarization and Transformer-question filtering, making 18 mapped benchmarks. No chunk IDs are defined yet.

Four explicit lexical/semantic contrasts paraphrase NLP, attendance/examination eligibility, Machine Learning and Computer Networks. Source phrasing must follow canonical descriptions without copying the query. Compare ranks empirically: semantic retrieval may help, but no winner is guaranteed. Numeric policy facts remain numeric in JSON. The networks query is intentionally open enough to discuss ambiguity.

The NLP examination/topics benchmark requires the calendar and syllabus together. The paper-frequency benchmark requires all four actual paper texts. The papers are explicitly undated synthetic archive-style sets, not real historical evidence. They do not invent prior academic years or copy the single scheduled NLP examination date onto every paper.

## Question-paper topic strategy

The taxonomy contains 16 topics, aliases and related-topic links. Aliases are matching cues, not infallible classification rules. Word Embeddings, Word2Vec, CBOW and Skip-Gram remain distinct; related links explain overlap.

Four paper IDs each have 12 top-level question slots. Each slot has exactly one primary topic for counting; secondary annotations do not add to the primary-topic frequency. The per-paper targets total 48 slots, with Transformers and TF-IDF most frequent, followed by embeddings and attention, then Word2Vec, BERT and preprocessing/tokenization. Other topics occur less often. Counts are stored once in `question_counts_by_paper`; aggregate frequencies are computed from them.

Each generated paper contains twelve compulsory five-mark questions, totaling 60 end-semester marks. There are no choice branches or unrelated sub-question slots. Stable Q1–Q12 headings make extraction straightforward. Questions vary across explanation, comparison, illustration, calculation and analysis. Some use aliases or implicit descriptions, so a topic classifier must handle more than exact term matching. Historical sitting dates and examination durations remain unspecified.

## Ground truth and application behaviour — permanent rule

`data/question_papers/question_metadata.json` contains authored text/mark alignment and primary/secondary topic labels for validation and accuracy evaluation. **The Question Paper Analyzer must not read this metadata or canonical target frequencies to produce user-facing analysis.** It must extract questions, identify topics, calculate counts and rank results independently from the Markdown paper text. Secondary topics support evaluation of related-topic filtering; they do not change the primary-topic distribution.

**Retrieval and RAG must search document content, not benchmark expected references or answers.** Tests may read ground truth. The manifest's content inventory should guide ingestion; `university_facts.json`, `question_metadata.json` and `retrieval_benchmarks.json` are validation/design artifacts, not answer passages to index. A question-paper analysis route reads the paper Markdown. Primary Transformer classification and broader related-question filtering are different evaluation tasks; the latter may also identify relevant BERT or attention questions with an explanation.

The existing canonical paper-frequency benchmark still references the target contract for validation. That reference is never a substitute for reading and analyzing paper text. It must not be exposed as a computed analyzer result.

## Task 03 contract additions

Sixteen selected courses lacked a compact description. Task 03 adds an original description for each, plus keyword lists for all 20 corpus courses, so the CSV derives from canonical fields rather than inventing course descriptions independently. The four previously approved descriptions are unchanged. Names, codes, semesters, credits, prerequisites, faculty, policy values, dates, taxonomy and per-paper target counts remain unchanged.

Generation-status fields now reflect generated Markdown and papers. The paper assessment-scope text records the twelve-compulsory-five-mark layout selected in Task 03. These resolve stale planning metadata; they do not alter the approved 60-mark component or topic distribution. No commercial services, extra dependencies or models are needed.

## Validation and change discipline

Task 04 refines NLP Unit 4 with GPT and autoregressive / next-token language modeling, contrasting decoder-oriented generation with BERT's encoder-oriented masked modeling and contextual representation. The existing `language_models` taxonomy entry supplies aliases and structured subtopic descriptions; no new primary topic ID or question-frequency target is introduced. The NLP Markdown syllabus follows this refinement. Existing course values, policies, dates, question papers and ground-truth counts are unchanged. This does not introduce GPT inference.

Run `python -m pytest` from the repository root. Tests cover the foundation, canonical contract and generated dataset. Dataset checks compare curriculum and prerequisite tables, syllabus identity and units, faculty assignments, attendance, marks, dates, CSV fields, extracted paper text, per-paper/aggregate primary-topic counts, manifest coverage and benchmark references. No NLP libraries or new runtime dependencies are needed.

Before NLP implementation, review the generated wording, question-topic annotation conventions and supplementary benchmark coverage. Subsequent fact changes must regenerate affected documents and update their tests and retrieval expectations together. Conceptual syllabus explanations are elaborations of approved topics, not additional institutional requirements. The input examples contain no solved vectors, scores or analyzer results.
