"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Exercise 6 pairwise similarity; loading/cache/display are starter code."""
    import csv
    from pathlib import Path
    import streamlit as st
    from src.embeddings import encode_sentences, sentence_similarity, pairwise_tfidf_similarity, SIMILARITY_PAIRS
    from src.document_processor import model_token_lengths
    from src.classical_nlp import cosine_similarity, build_query_tfidf_vector
    from src.retrieval import prepare_search_corpus, tokenize_for_search
    from src.sentence_resources import SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION, SentenceModelUnavailable, load_sentence_embedding_model

    ROOT = Path(__file__).resolve().parents[1]
    st.set_page_config(page_title="Sentence Embeddings Lab", page_icon="🧭")
    st.title("Sentence Embeddings and Semantic Similarity")
    st.write("Exercise 6 · Sentence → encoder → dense vector → manual cosine → interpretation")
    st.subheader("1 — From word vectors to sentence vectors")
    st.write("Word2Vec gives each vocabulary word one static vector. This encoder gives one vector per complete text using context and composition. It is not an average of our Word2Vec vectors.")
    st.subheader("2 — Shared sentence encoder")
    st.code(SENTENCE_MODEL_ID)
    st.write("Free pretrained Transformer-based representations, local CPU inference. We use learned representations rather than train an encoder on our small dataset. Transformer internals come in a later lab.")
    st.caption("Preload while online: python -m src.sentence_resources --download. Verify offline: python -m src.sentence_resources. Normal page loading never downloads a model.")


    @st.cache_resource(max_entries=1)
    def cached_encoder(revision):
        if revision != SENTENCE_MODEL_REVISION: raise ValueError("Use the pinned workshop encoder.")
        return load_sentence_embedding_model()


    @st.cache_data(max_entries=64)
    def cached_vectors(texts, revision):
        return encode_sentences(cached_encoder(revision), list(texts))


    @st.cache_data(max_entries=64)
    def cached_similarity(first, second, revision):
        return sentence_similarity(cached_encoder(revision), first, second)


    if st.button("Reload encoder after setup or source edits", key="sentence_reload"):
        cached_encoder.clear()
        cached_vectors.clear()
        cached_similarity.clear()
        from src.sentence_resources import _load_cached_model
        _load_cached_model.cache_clear()
    try:
        with st.spinner("Loading the locally cached CPU encoder…"):
            model = cached_encoder(SENTENCE_MODEL_REVISION)
    except SentenceModelUnavailable as error:
        st.warning("The sentence model is not ready. Other workshop labs remain available.")
        st.info(str(error))
        st.stop()

    st.caption("384 dimensions · original text retained · no extra normalization requested. The model's own Normalize module is preserved, so norms are already about 1. Cosine is calculated explicitly.")
    st.info("Similarity is not a probability, factual check, or guarantee of relevance. Text beyond 256 word pieces is truncated by this encoder.")
    st.subheader("3 — Encode a complete sentence")
    text = st.text_area("Sentence to inspect", "Natural language processing helps computers understand human language.", key="sentence_inspect", max_chars=8000)
    if text.strip():
        pieces = model_token_lengths(model.tokenizer, [text])[0]
        if pieces > model.max_seq_length:
            st.warning(f"This input has {pieces} model tokens; only the first {model.max_seq_length} enter the encoder. Split long passages into chunks in the RAG Lab.")
    if text.strip():
        try:
            vector = cached_vectors((text,), SENTENCE_MODEL_REVISION)[0].tolist()
            norm = cosine_similarity(vector, vector, explain=True)["magnitude_a"]
            a, b, c = st.columns(3)
            a.metric("Number of sentences", 1)
            b.metric("Embedding shape", "(1, 384)")
            c.metric("Vector norm", f"{norm:.6f}")
            st.dataframe([{"Dimension": i, "Value": value} for i, value in enumerate(vector[:10])], hide_index=True)
            st.caption("First ten dimensions only; coordinates do not have named human meanings.")
        except (ValueError, RuntimeError) as error:
            st.warning(f"Encoding unavailable: {error}")
    else:
        st.info("Enter non-empty text to inspect a vector.")

    st.subheader("4 — Compare two sentences")
    sentence_a = st.text_area("Sentence A", "Natural language processing helps computers understand human language.", key="sentence_a")
    sentence_b = st.text_area("Sentence B", "Machines can analyze the meaning of human communication.", key="sentence_b")
    if sentence_a.strip() and sentence_b.strip():
        try:
            pair_lengths = model_token_lengths(model.tokenizer, [sentence_a, sentence_b])
            if max(pair_lengths) > model.max_seq_length:
                st.warning(f"Pair token lengths {pair_lengths}; the encoder uses at most {model.max_seq_length} per sentence. Similarity can omit later content.")
            vectors = cached_vectors((sentence_a, sentence_b), SENTENCE_MODEL_REVISION)
            st.write("Sentence A → embedding row 0; Sentence B → embedding row 1")
            st.dataframe([{"Sentence": label, "Dimensions": len(vector), "First values": ", ".join(f"{value:.4f}" for value in vector[:5])}
                          for label, vector in zip(["A", "B"], vectors)], hide_index=True)
            st.metric("Manual cosine over sentence embeddings", f"{cached_similarity(sentence_a, sentence_b, SENTENCE_MODEL_REVISION):.6f}")
            st.json(cosine_similarity(vectors[0].tolist(), vectors[1].tolist(), explain=True))
            st.write("Custom-pair TF-IDF cosine:", round(pairwise_tfidf_similarity(sentence_a, sentence_b), 6))
        except (ValueError, RuntimeError) as error:
            st.warning(f"Cannot compare these texts: {error}")
    else:
        st.info("Both comparison sentences must contain text; blank strings are rejected.")

    st.subheader("5 — Similarity experiments")
    st.write("Predict first, then inspect measured values. Reversing roles can fool both representations.")
    with st.expander("Preset text pairs", expanded=True):
        for label, first, second in SIMILARITY_PAIRS:
            st.markdown(f"**{label}**\n\nA: {first}\n\nB: {second}")
    st.subheader("6 — TF-IDF vs sentence embeddings (pairwise)")
    preset_texts = tuple(text for _, first, second in SIMILARITY_PAIRS for text in (first, second))
    try:
        preset_vectors = cached_vectors(preset_texts, SENTENCE_MODEL_REVISION)
        st.dataframe([{"Pair": label, "TF-IDF cosine": pairwise_tfidf_similarity(first, second),
                       "Sentence-embedding cosine": cosine_similarity(preset_vectors[2*i].tolist(), preset_vectors[2*i+1].tolist())}
                      for i, (label, first, second) in enumerate(SIMILARITY_PAIRS)], hide_index=True)
    except (ValueError, RuntimeError) as error:
        st.warning(f"Preset encoding unavailable: {error}")
    st.caption("Classroom TF=count/length; IDF=ln(N/DF). Fit on eight preset texts plus distinct custom inputs. A two-text-only fit gives every shared term IDF zero, so this visible reference collection is used. Scores are not the fixed 20-course baseline below.")

    st.subheader("7 — Task 06 human-communication failure case")
    query = "Which subject teaches machines to work with human communication?"
    st.write("Query:", query)
    with (ROOT / "data/search_corpus/course_descriptions.csv").open(encoding="utf-8", newline="") as file:
        courses = list(csv.DictReader(file))
    # Full corpus is used only for the existing lexical IDF, never semantic ranking.
    lexical = prepare_search_corpus([{"document_id": course["course_code"], "text": course["description"]} for course in courses])
    query_vector = build_query_tfidf_vector(tokenize_for_search(query), lexical["vocabulary"], lexical["idf"])
    selected = [course for name in ["Natural Language Processing", "Distributed Systems"] for course in courses if course["course_name"] == name]
    try:
        examples = cached_vectors(tuple([query] + [course["description"] for course in selected]), SENTENCE_MODEL_REVISION)
        rows = []
        for i, course in enumerate(selected):
            rows.append({"Course": course["course_name"], "Task 06 TF-IDF cosine": cosine_similarity(query_vector, lexical["matrix"][courses.index(course)]),
                         "Sentence-embedding cosine": cosine_similarity(examples[0].tolist(), examples[i+1].tolist())})
            st.markdown(f"**{course['course_name']}** — {course['description']}")
        st.dataframe(rows, hide_index=True)
    except (ValueError, RuntimeError) as error:
        st.warning(f"Course-pair encoding unavailable: {error}")
    st.caption("Two manually selected descriptions only: Task 06 ranked NLP second (0.172652), behind Distributed Systems (0.313492). No semantic top-K or full corpus ranking is implemented here.")
    st.subheader("8 — What changed?")
    st.table([{"Representation": "TF-IDF", "Unit": "Document/query", "Space": "Vocabulary-frequency coordinates"},
              {"Representation": "Word2Vec", "Unit": "Word", "Space": "Static learned word coordinates"},
              {"Representation": "Sentence Transformer", "Unit": "Complete text", "Space": "Context-composed dense sentence coordinates"}])
    st.write("Cosine did not change; representation changed. High similarity does not verify facts, entailment, negation, or role direction.")
    st.subheader("9 — Next question")
    st.write("If every course description can become a sentence embedding, can we embed a user's query and retrieve the closest descriptions? That is the next lab: Semantic Search.")
    st.success("Exercise 6 milestone — Semantic Sentence Similarity Works. Overall Checkpoint5 is Transformer experiments.")

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
