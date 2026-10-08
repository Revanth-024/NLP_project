"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Exercise 7: fair description-only lexical/semantic comparison; starter UI."""
    import csv
    from io import StringIO
    from pathlib import Path
    import streamlit as st
    from src.retrieval import (prepare_search_corpus, search_tfidf, encode_corpus_documents,
                               semantic_search)
    from src.sentence_resources import (load_sentence_embedding_model, SentenceModelUnavailable,
                                        SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION)

    ROOT = Path(__file__).resolve().parents[1]
    PRESETS = {
        "Lexical — natural language processing": "natural language processing",
        "Lexical — machine learning": "machine learning",
        "Lexical — database systems": "database systems",
        "Lexical — computer communication networks": "computer communication networks",
        "Paraphrase — human communication": "Which subject teaches machines to work with human communication?",
        "Benchmark RET-03 — interpreting language": "Which subject teaches computers to interpret human language?",
        "Benchmark RET-04 — classroom presence": "How much classroom presence do I need to sit my finals?",
        "Benchmark RET-05 — prediction from examples": "Which class learns to predict outcomes from examples?",
        "Benchmark RET-06 — connected devices": "Where do connected devices learn the rules for exchanging messages?",
        "Ambiguous — learning": "learning",
        "Unrelated — baking": "How do I bake a chocolate cake?",
        "Custom query": None,
    }

    st.set_page_config(page_title="Semantic Search Lab", page_icon="🔎", layout="wide")
    st.title("Semantic Search — TF-IDF vs Sentence Embeddings")
    st.write("Exercise 7 · Documents → embeddings; query → same encoder → cosine with every row → rank → Top-K")
    st.subheader("1 — From pairwise similarity to search")
    st.write("We changed the representation, not cosine, ranking or Top-K. A query is now compared with every course description. Retrieval selects evidence; it does not generate an answer.")
    st.info("Both methods index descriptions only. Course titles, codes and other metadata are labels, never hidden boosts. Cosine scores are not probabilities and the two methods are not identically calibrated; compare relevance and ranks.")


    @st.cache_resource(max_entries=1)
    def cached_encoder(revision):
        if revision != SENTENCE_MODEL_REVISION: raise ValueError("Use the pinned workshop encoder.")
        return load_sentence_embedding_model()


    @st.cache_data(max_entries=4)
    def cached_document_embeddings(documents, model_id, revision):
        # The text/order and exact model/revision are part of the cache key.
        return encode_corpus_documents(cached_encoder(SENTENCE_MODEL_REVISION), documents)


    @st.cache_data(max_entries=4)
    def cached_lexical_corpus(documents):
        return prepare_search_corpus(documents)


    @st.cache_data(max_entries=4)
    def cached_ambiguity_examples(documents, revision):
        matrix = cached_document_embeddings(documents, SENTENCE_MODEL_ID, revision)
        lexical = cached_lexical_corpus(documents)
        rows = []
        for text in ["learning", "How do I bake a chocolate cake?"]:
            semantic = semantic_search(text, documents, matrix, cached_encoder(SENTENCE_MODEL_REVISION), 3)
            classical = search_tfidf(text, lexical, 3)
            rows.append({"Query": text, "TF-IDF Top-1": classical["results"][0]["title"],
                         "TF-IDF cosine": classical["results"][0]["score"],
                         "Semantic Top-1": semantic["results"][0]["title"],
                         "Semantic cosine": semantic["results"][0]["score"]})
        return rows


    st.subheader("2 — Same academic course corpus")
    csv_text = (ROOT / "data/search_corpus/course_descriptions.csv").read_text(encoding="utf-8")
    courses = list(csv.DictReader(StringIO(csv_text)))
    documents = [{"document_id": row["course_code"], "title": row["course_name"], "text": row["description"],
                  "semester": int(row["semester"]), "credits": int(row["credits"])} for row in courses]
    with st.expander("Inspect all twenty descriptions and labels"):
        st.dataframe(courses, hide_index=True)
    st.caption(f"Shared CPU encoder: {SENTENCE_MODEL_ID} · pinned revision {SENTENCE_MODEL_REVISION}")
    if st.button("Clear retrieval caches after source edits", key="semantic_clear"):
        cached_document_embeddings.clear()
        cached_lexical_corpus.clear()
        cached_encoder.clear()
        cached_ambiguity_examples.clear()
    try:
        with st.spinner("Loading the local encoder and preparing cached corpus vectors…"):
            model = cached_encoder(SENTENCE_MODEL_REVISION)
            document_embeddings = cached_document_embeddings(documents, SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION)
    except (SentenceModelUnavailable, ValueError, RuntimeError) as error:
        st.warning("Semantic search is not ready. Other labs remain available.")
        st.info(str(error))
        st.caption("Preload explicitly: python -m src.sentence_resources --download; verify locally: python -m src.sentence_resources. This page never downloads a model.")
        st.stop()
    lexical_corpus = cached_lexical_corpus(documents)
    st.subheader("3 — Corpus embedding matrix")
    first, second, third = st.columns(3)
    first.metric("Documents", len(documents))
    second.metric("Dimensions per document", document_embeddings.shape[1])
    third.metric("Matrix shape", str(document_embeddings.shape))
    st.caption("Row i belongs to course i in the CSV. Original text is retained for MiniLM. Its native Normalize layer is preserved; no additional normalization is requested. Corpus vectors are reused when only query/Top-K changes.")


    def apply_preset():
        selected = PRESETS[st.session_state.semantic_preset]
        if selected is not None:
            st.session_state.semantic_query = selected


    st.subheader("4 — Query and Top-K")
    if "semantic_query" not in st.session_state:
        st.session_state.semantic_query = PRESETS["Paraphrase — human communication"]
    st.selectbox("Query experiment", list(PRESETS), index=4, key="semantic_preset", on_change=apply_preset)
    query = st.text_area("Search query", key="semantic_query")
    top_k = st.slider("Top-K for both methods", 1, len(documents), 5, key="semantic_top_k")
    if not query.strip():
        st.info("Enter a non-empty query before searching. Blank text is not meaningful evidence.")
        st.stop()
    try:
        semantic = semantic_search(query, documents, document_embeddings, model, top_k)
        lexical = search_tfidf(query, lexical_corpus, top_k)
    except (ValueError, RuntimeError) as error:
        st.warning(f"Cannot run this query: {error}")
        st.stop()

    st.subheader("5 — Query embedding")
    st.write("Shape: (384,) · same encoder and vector space as the corpus")
    st.dataframe([{"Dimension": i, "Value": value} for i, value in enumerate(semantic["query_vector"][:10])], hide_index=True)
    with st.expander("Lexical query tokens and OOV terms"):
        st.write("Tokens:", lexical["tokens"])
        st.write("Outside corpus vocabulary:", lexical["oov_terms"])
    st.subheader("6 — Similarity vector → sorted indices")
    with st.expander("Every score in original corpus order", expanded=True):
        st.dataframe([{"CSV index": i, "Course": document["title"], "TF-IDF cosine": lexical["scores"][i],
                       "Semantic cosine": semantic["scores"][i]} for i, document in enumerate(documents)], hide_index=True)
    st.write("Sorted semantic indices (zero-based):")
    st.code(str(semantic["sorted_indices"]), language=None)

    st.subheader("7–8 — Top-K results and fair comparison")
    left, right = st.columns(2)
    for column, heading, result in [(left, "TF-IDF search", lexical), (right, "Semantic search", semantic)]:
        with column:
            st.markdown(f"### {heading}")
            for rank, row in enumerate(result["results"], 1):
                st.markdown(f"**{rank}. {row['title']}** · Similarity: {row['score']:.6f}")
                st.write(row["text"])
                if heading == "TF-IDF search":
                    position = next(i for i, document in enumerate(documents)
                                    if document["document_id"] == row["document_id"])
                    document_terms = set(lexical_corpus["tokens"][position])
                    matched = sorted(set(lexical["tokens"]) & document_terms)
                    st.caption("Shared lexical terms: " + (", ".join(matched) or "none"))
                else:
                    st.caption("This description's embedding is close to the query embedding; no token-level attribution is claimed.")
    st.dataframe([{"Rank": i+1, "TF-IDF result": lexical["results"][i]["title"], "TF-IDF score": lexical["results"][i]["score"],
                   "Semantic result": semantic["results"][i]["title"], "Semantic score": semantic["results"][i]["score"]}
                  for i in range(len(semantic["results"]))], hide_index=True)

    with st.expander("Inspect rank changes for a selected course"):
        selected_course = st.selectbox("Course to track (your hypothesis, not a benchmark answer)", [d["title"] for d in documents], index=next(i for i,d in enumerate(documents) if d["title"]=="Natural Language Processing"), key="semantic_track")
        position = next(i for i,d in enumerate(documents) if d["title"]==selected_course)
        # Ranking is unchanged: stable original-order tie handling for both methods.
        from src.retrieval import rank_documents
        lexical_all = rank_documents(documents, lexical["scores"], len(documents))
        lexical_rank = next(i for i,row in enumerate(lexical_all,1) if row["title"]==selected_course)
        semantic_rank = semantic["sorted_indices"].index(position)+1
        st.write(f"TF-IDF rank: {lexical_rank}; semantic rank: {semantic_rank}; rank change (TF-IDF minus semantic): {lexical_rank-semantic_rank:+d}")
        st.caption("Positive means the selected course moved up; negative means it moved down. This does not by itself establish relevance.")

    st.subheader("9 — Benchmark experiments")
    st.write("The RET-03–RET-06 presets reuse existing wording. This page searches only course descriptions: policy/calendar/paper evidence is outside this corpus. Full-source benchmark evaluation is instructor tooling, not search output.")
    st.code("python -m workshop.evaluate_retrieval")
    st.caption("The evaluator reads expected IDs only after searching content. Its separate measured report includes all eighteen unchanged benchmarks and discloses long-document truncation. The application does not read expected answers or the measured report.")
    st.subheader("10 — Failures, ambiguity and score interpretation")
    st.dataframe(cached_ambiguity_examples(documents, SENTENCE_MODEL_REVISION), hide_index=True)
    st.write("The baking query has no relevant course: semantic search still returns a nearest vector, while TF-IDF's all-zero scores still produce an original-order tie. Neither result establishes relevance. Learning is ambiguous across several courses. Try these presets and inspect all scores. The instructor guide separately records full-source benchmark regressions and improvements, including the human-communication query's full ranking.")
    st.subheader("11 — End of Day 1; next question")
    st.write("We can retrieve information. Can we use retrieved information to answer a question? Day 2 first introduces Transformers and document processing: full academic documents need meaningful retrieval units before generation.")
    st.success("Overall Checkpoint4 — Semantic Search Works. Continue to Transformers and the RAG Lab for document chunk retrieval.")

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
