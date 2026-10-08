# Natural Language Processing Syllabus

**Synthetic educational data created for NLP workshop purposes.**

Hindu College of Engineering is a completely fictional educational institution created only for this workshop.

## Course information

| Field | Value |
| --- | --- |
| Course code | HCE-CSE603 |
| Course name | Natural Language Processing |
| Semester | 6 |
| Credits | 4 |
| Delivery | theory |
| Faculty | Dr. Nivora Pellin |
| Academic year | 2026-2027 |

## Course description

Computational methods for processing, understanding and generating human language through symbolic, statistical and contextual representations.

## Prerequisites

Earlier-course prerequisites: Artificial Intelligence (HCE-CSE501); Machine Learning (HCE-CSE502).

The related course is Natural Language Processing Laboratory (HCE-CSE605), a 2-credit laboratory in Semester 6. The theory course is its same-semester corequisite, not an earlier prerequisite. The laboratory's own earlier prerequisites are Artificial Intelligence (HCE-CSE501); Machine Learning (HCE-CSE502).

## Learning objectives

- Prepare text and explain representation choices.
- Construct count-based and weighted language representations.
- Compare distributed word representations.
- Explain sequence context, attention and Transformer representations.
- Evaluate semantic similarity and retrieval experiments.

## Unit structure

### Unit 1 — Text Preparation

Topics: Text normalization; Stop words; Stemming and lemmatization; Text Preprocessing; Tokenization.

Text preparation begins with the decisions that turn a raw string into a usable sequence. Students consider capitalization, punctuation and spacing alongside text normalization. Tokenization establishes the units that later representations count or encode. Stop words, stemming and lemmatization are examined as choices with consequences rather than operations that should be applied automatically. An abbreviated course notice and a complete sentence can lose different information when cleaned in the same way. The unit connects visible changes in the text to the task being attempted.

### Unit 2 — Count-Based Language Representation

Topics: Vocabulary; One-hot encoding; Term and document frequency; Bag of Words; N-Grams; TF-IDF; Cosine Similarity.

A vocabulary supplies a shared coordinate system for a collection. One-hot encoding identifies individual terms; Bag of Words counts their occurrences while setting aside order. N-Grams retain short sequences and therefore preserve some local context. Term frequency and document frequency support TF-IDF, where frequent words within one document are interpreted alongside their prevalence across the collection. Cosine Similarity compares vector direction rather than simply rewarding a long document. Students discuss what a count-based representation can capture and what it loses when two passages use different words for related ideas.

### Unit 3 — Distributed Word Representation

Topics: Embedding geometry; Distributional context; Word vector evaluation; Word Embeddings; Word2Vec; CBOW; Skip-Gram.

Distributed word representations place vocabulary items in a dense numerical space rather than allocating an independent coordinate to every word. The unit connects Word Embeddings with patterns of distributional context. Word2Vec is considered through its two training directions: CBOW uses surrounding context to predict a target, while Skip-Gram starts with a target to predict nearby words. Vector geometry supports comparisons among learned words, but closeness does not automatically prove synonymy. Evaluation asks whether a representation captures useful relationships for a particular language task and where a single representation per word misses contextual variation.

### Unit 4 — Sequence and Contextual Models

Topics: Sequence context; Recurrent model intuition; Contextual representations; Masked language modeling; Autoregressive / next-token language modeling; GPT; Language Models; Attention; Transformers; BERT.

Sequence models introduce the importance of surrounding tokens and their order. Recurrent model intuition provides a point of comparison for contextual representations. Attention assigns differing relevance to parts of a sequence when constructing a representation; Transformers organize these operations into an architecture that can relate distant positions. BERT introduces an encoder-oriented contextual representation and masked language modeling. GPT provides the contrasting decoder-oriented, autoregressive / next-token language modeling approach used for text generation: successive tokens are predicted from preceding context. The unit distinguishes an architecture from a training objective and a pretrained model family. Language Models connect token prediction with the use of context, without assuming that fluent text guarantees factual correctness.

### Unit 5 — Meaning and Retrieval

Topics: Sentence representations; Retrieval evaluation; Limitations of similarity measures; Semantic Similarity; Semantic Search.

Sentence representations extend the discussion from isolated vocabulary items to complete pieces of text. Semantic Similarity asks how closely two statements relate in meaning, while Semantic Search uses such representations to retrieve relevant evidence from a collection. Students connect similarity measures with retrieval evaluation and inspect ambiguous matches rather than treating a high score as a complete explanation. A query may use terminology absent from a relevant passage. Conversely, overlapping words may refer to different academic topics. The unit emphasizes identifying evidence, explaining ranking behaviour and stating the limitations of the chosen representation.

## Expected learning outcomes

Students should be able to explain how a representation changes the information available to a language task, compare count-based and distributed approaches, and interpret semantic retrieval evidence with appropriate limitations. The course connects Machine Learning methods with language applications. Retrieval-Augmented Generation is a subsequent workshop application, not a sixth formal syllabus unit.

## Assessment reference

The course has 100 total marks: 40 internal assessment marks and 60 end-semester marks. Passing requires at least 50 total marks and at least 24 end-semester marks. Both conditions apply together; there is no separate internal minimum. Attendance eligibility remains a separate requirement. Internal assessment contains a 20-mark midterm and 20 marks of continuous coursework. Written end-semester assessment is used for this theory course. The associated laboratory uses the same marking scheme with practical demonstration and viva. Refer to Examination Regulations for grade bands and Attendance Policy for eligibility; no additional course-specific pass threshold is introduced here.
