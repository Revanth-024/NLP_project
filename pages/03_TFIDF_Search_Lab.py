"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Starter-provided visualization for classical TF-IDF and search exercises."""
    import csv
    from pathlib import Path
    import streamlit as st
    from src.classical_nlp import (
        build_document_term_matrix, calculate_tf, sklearn_tfidf_comparison,
        build_query_tfidf_vector, cosine_similarity,
    )
    from src.retrieval import prepare_search_corpus, search_tfidf, score_documents, rank_documents

    st.set_page_config(page_title="TF-IDF Search Lab", page_icon="🔎")
    st.title("Exercises 3 and 4 — TF-IDF and First Search Engine")
    st.caption("Student starter state: boundaries 3.1–3.5 and 4.1–4.3. All calculations live in shared src modules.")
    st.write("Documents → Vocabulary → TF → DF → IDF → TF-IDF → Query Vector → Cosine → Ranked Results")
    st.write("Are all words equally informative? Once words become vectors, how can we compare a user's query with documents?")
    root = Path(__file__).resolve().parents[1]
    choice = st.radio("Corpus", ["Learning Corpus", "Academic Course Corpus"], key="tfidf_corpus")
    if choice == "Learning Corpus":
        raw = st.text_area("One learning document per line", value="Natural language processing analyzes human language.\nMachine learning discovers patterns from data.\nComputer networks connect devices and exchange information.\nDatabase systems organize and query structured data.", key="tfidf_documents", height=150, max_chars=10000)
        if len(raw) > 10000 or len(raw.splitlines()) > 20:
            st.info("Use up to 20 learning documents / 10,000 characters for an inspectable matrix."); st.stop()
        documents = [{"document_id": f"D{i + 1}", "title": f"Learning document {i + 1}", "text": text.strip()} for i, text in enumerate(raw.splitlines()) if text.strip()]
    else:
        with (root / "data/search_corpus/course_descriptions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        documents = [{"document_id": row["course_code"], "title": row["course_name"], "text": row["description"]} for row in rows]
        st.caption("Search the 20 supplied fictional course descriptions. Titles label results; only descriptions are indexed, without keyword-field boosting.")
    st.caption("Shared policy: lowercase → punctuation replaced by spaces → tokenization. Stopwords and word forms are retained. No models or corpus downloads.")
    if not documents:
        st.info("Enter at least one document to inspect weights and search.")
        st.stop()
    corpus = prepare_search_corpus(documents)
    vocabulary = corpus["vocabulary"]
    if not vocabulary:
        st.info("No tokens remain. Enter document words rather than punctuation alone.")
        st.stop()
    st.subheader("1. Documents, tokens and shared vocabulary")
    for document, tokens in zip(documents, corpus["tokens"]):
        with st.expander(document["document_id"] + " — " + document["title"]):
            st.write(document["text"])
            st.json(tokens)
    st.metric("Vocabulary dimensions", len(vocabulary))
    st.json(vocabulary)
    st.subheader("2. BoW → TF → IDF → TF-IDF")
    bow = build_document_term_matrix(corpus["tokens"], vocabulary)
    tf_matrix = [calculate_tf(tokens, vocabulary) for tokens in corpus["tokens"]]
    with st.expander("Inspect labelled matrices and rarity values", expanded=choice == "Learning Corpus"):
        for label, matrix in [("BoW counts", bow), ("TF = count / document length", tf_matrix), ("Manual TF-IDF = TF × IDF", corpus["matrix"])]:
            st.write(label)
            st.dataframe([{"Document": document["document_id"], **dict(zip(vocabulary, row))} for document, row in zip(documents, matrix)], hide_index=True)
        st.write("DF counts documents. IDF uses natural logarithm: ln(N / DF).")
        st.dataframe([{"Term": term, "DF": df, "IDF": idf} for term, df, idf in zip(vocabulary, corpus["df"], corpus["idf"])], hide_index=True)
    st.subheader("3. TF-IDF calculation explorer")
    selected_document = st.selectbox("Document to inspect", range(len(documents)), format_func=lambda i: documents[i]["document_id"] + " — " + documents[i]["title"], key="tfidf_doc")
    term = st.selectbox("Vocabulary term", vocabulary, key="tfidf_term")
    column = vocabulary.index(term)
    st.table([{"Term Count": bow[selected_document][column], "Document Length": len(corpus["tokens"][selected_document]), "TF": tf_matrix[selected_document][column], "DF": corpus["df"][column], "N": len(documents), "IDF": corpus["idf"][column], "TF-IDF": corpus["matrix"][selected_document][column]}])
    st.caption("An empty document has TF zero. Terms appearing in every document have IDF zero under this classroom convention.")
    with st.expander("Common versus rare: a hand-calculable example"):
        demo = prepare_search_corpus([{"document_id": "A", "text": "common nlp nlp"}, {"document_id": "B", "text": "common text"}, {"document_id": "C", "text": "common text"}])
        st.table([{"Term": term, "DF": df, "IDF": idf} for term, df, idf in zip(demo["vocabulary"], demo["df"], demo["idf"])])
        st.write("common: DF=3, IDF=0; text: DF=2, IDF≈0.4055; nlp: DF=1, IDF≈1.0986. Rare does not automatically mean relevant.")
    st.subheader("4. Query → vector → cosine → ranking")
    st.write("Try: natural language processing; machine learning; database systems; computer communication networks; Which subject teaches machines to work with human communication?")
    query = st.text_input("Search query", value="natural language processing", key="tfidf_query")
    top_k = st.slider("Top K", 1, len(documents), min(3, len(documents)), key="tfidf_top_k") if len(documents) > 1 else 1
    search = search_tfidf(query, corpus, top_k)
    st.write("Query tokens")
    st.json(search["tokens"])
    st.write("Out-of-vocabulary terms (ignored)")
    st.json(search["oov_terms"])
    st.caption("Reuse the corpus vocabulary and IDF. Query TF is normalized over retained known tokens; ignored terms create no dimensions. BoW teaching functions remain strict, but search accepts novel query terms.")
    st.dataframe([{"Term": term, "Query TF-IDF": value} for term, value in zip(vocabulary, search["query_vector"])], hide_index=True)
    if not any(search["query_vector"]):
        st.warning("Query vector is zero: no positive-weight vocabulary overlap. Empty/OOV queries or corpus-wide terms can cause this. All scores are zero; tied rows retain original order and are not relevant matches.")
    st.write("All similarity scores in original corpus order")
    st.table([{"Document": document["document_id"], "Cosine": score} for document, score in zip(documents, search["scores"])])
    q = search["query_vector"]
    d = corpus["matrix"][selected_document]
    calculation = cosine_similarity(q, d, explain=True)
    st.table([{"Compared document": documents[selected_document]["document_id"], "Dot product": calculation["dot_product"], "Query magnitude": calculation["magnitude_a"], "Document magnitude": calculation["magnitude_b"], "Denominator": calculation["denominator"], "Cosine": calculation["similarity"]}])
    st.caption("Cosine compares direction rather than raw length; it is a similarity score, not a probability. Zero denominator returns 0.0.")
    st.subheader("5. Ranked search results")
    known = set(search["tokens"]) & set(vocabulary)
    for rank, result in enumerate(search["results"], 1):
        position = next(i for i, document in enumerate(documents) if document["document_id"] == result["document_id"])
        st.write(f"{rank}. {result['title']} ({result['document_id']}) — cosine {result['score']:.4f}")
        st.write(result["text"])
        st.caption("Matched vocabulary terms: " + (", ".join(sorted(known & set(corpus["tokens"][position]))) or "none"))
    st.subheader("6. Manual first, sklearn comparison second")
    library = sklearn_tfidf_comparison(corpus["tokens"], vocabulary)
    st.write("Manual: TF=count/length; IDF=ln(N/DF). sklearn: raw counts × [ln((1+N)/(1+DF))+1], smooth_idf=True, sublinear_tf=False, norm=None. Same tokens and fixed vocabulary; matrices are not expected to be equal. sklearn's default norm is L2; disabled here to inspect raw weights.")
    with st.expander("Inspect sklearn weights and ranking"):
        st.json(vocabulary)
        st.dataframe([{"Term": term, "Manual IDF": manual, "sklearn IDF": automatic} for term, manual, automatic in zip(vocabulary, corpus["idf"], library["idf"])], hide_index=True)
        st.dataframe([{"Document": document["document_id"], **dict(zip(vocabulary, row))} for document, row in zip(documents, library["matrix"])], hide_index=True)
        # Normalized query TF is a positive row scaling of sklearn raw counts;
        # cosine is unchanged by that scale, with sklearn's corpus IDF retained.
        library_query = build_query_tfidf_vector(search["tokens"], vocabulary, library["idf"])
        library_scores = score_documents(library_query, library["matrix"])
        st.table([{"Document": result["document_id"], "sklearn cosine": result["score"]} for result in rank_documents(documents, library_scores, top_k)])
        st.caption("Query uses sklearn corpus IDF; normalizing its counts only rescales the vector and leaves cosine unchanged. Different IDF can change rankings. We do not fit IDF on the query.")
    st.success("Checkpoint 3: We built a search engine without an LLM.")
    st.write("Lexical overlap is not semantic understanding. Try the human-communication paraphrase and inspect matched terms. Word embeddings will address meaning in later exercises.")

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
