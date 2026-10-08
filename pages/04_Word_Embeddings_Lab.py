"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Starter UI/cache/PCA infrastructure around shared Exercise 5 functions."""
    import json
    from pathlib import Path
    import streamlit as st
    from src.classical_nlp import one_hot_encode, cosine_similarity
    from src.embeddings import (
        DEFAULT_WORD2VEC, TRAINING_SETTINGS, prepare_tokenized_sentences,
        train_word2vec, get_word_vector, find_similar_words, calculate_word_similarity,
        project_word_vectors,
    )

    st.set_page_config(page_title="Word Embeddings Lab", page_icon="📚")
    st.title("Exercise 5 — Word Embeddings and Word2Vec")
    st.caption("Student exercise cores 5.1–5.5 are in src/embeddings.py. UI, caching and PCA are provided.")
    st.warning("This Word2Vec model is intentionally trained on a small workshop corpus. Its similarities are demonstrations of representation learning, not production-quality semantic judgments.")
    st.subheader("1. Why move beyond one-hot and TF-IDF?")
    st.write("Task 06's human-communication paraphrase ranked Distributed Systems above NLP in description-only lexical search. Can context teach related representations? Words occurring in similar contexts tend to develop related representations: the distributional hypothesis.")
    st.write("One-hot has vocabulary-sized independent sparse dimensions. Word2Vec learns dense coordinates from context; individual dimensions do not have simple labels such as technical or academic.")
    root = Path(__file__).resolve().parents[1]
    try:
        data = json.loads((root / "data/examples/word2vec_corpus.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        st.warning("The workshop Word2Vec corpus is unavailable. Restore data/examples/word2vec_corpus.json and run python -m workshop.preflight.")
        st.caption(str(error)); st.stop()
    sentences = [item["text"] for item in data["sentences"]]
    tokens = prepare_tokenized_sentences(sentences)
    st.subheader("2. Inspect the training corpus")
    st.caption(f"{len(tokens)} sentence-like spans derived from {len(data['source_files'])} existing synthetic sources. No benchmark answers, metadata labels or external text.")
    with st.expander("Corpus provenance and tokenized sentences"):
        st.write(data["disclaimer"])
        st.write("Hindu College of Engineering is fictional.")
        st.write(data["derivation"])
        sample_count = st.slider("Sentences to inspect", 5, len(tokens), 10, key="w2v_sample_count")
        for item, sentence_tokens in zip(data["sentences"][:sample_count], tokens[:sample_count]):
            st.caption(item["source"])
            st.write(item["text"])
            st.json(sentence_tokens)
    st.subheader("3. Train Word2Vec locally")
    architecture = st.radio("Architecture", ["CBOW", "Skip-Gram"], key="w2v_architecture")
    vector_size = st.selectbox("Vector size", [20, 50, 100], index=1, key="w2v_vector_size")
    window = st.slider("Context window", 1, 8, DEFAULT_WORD2VEC["window"], key="w2v_window")
    st.caption("CBOW: context → target (natural ___ processing → language). Skip-Gram: target → nearby context (language → natural, processing). Gensim trains the network; students configure the workflow.")
    with st.expander("Fixed advanced settings"):
        st.json({**DEFAULT_WORD2VEC, **TRAINING_SETTINGS, "sg": 0 if architecture == "CBOW" else 1, "vector_size": vector_size, "window": window})
        st.caption("Stable SHA-256 initialization hash is starter infrastructure. Fixed seed and workers=1 improve repeatability; exact cross-platform floats are not guaranteed.")
    st.caption("Smaller windows can emphasize local relationships; larger windows can emphasize broader topics. Larger vectors have capacity but need more data; noisy outcomes are expected here. These are intuitions, not guaranteed rankings.")

    @st.cache_resource(max_entries=6, show_spinner="Training tiny CPU Word2Vec model…")
    def cached_model(sentence_tokens, size, context_window, sg):
        # Entire corpus and settings are cache keys; no excluded underscore argument.
        return train_word2vec(sentence_tokens, vector_size=size, window=context_window, sg=sg)

    if st.button("Clear cached models and retrain", key="w2v_retrain"):
        cached_model.clear()
    st.caption("After editing the training core, use this provided control to rebuild cached models.")

    sg = 0 if architecture == "CBOW" else 1
    try:
        model = cached_model(tokens, vector_size, window, sg)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    st.caption("Models are cached in memory for identical corpus/settings; word selections do not retrain them. No pretrained vectors are downloaded or model files saved.")
    vocabulary = sorted(model.wv.index_to_key)
    st.metric("Learned vocabulary size", len(vocabulary))
    st.subheader("4. Inspect a word vector")
    word = st.selectbox("Vocabulary word", vocabulary, index=vocabulary.index("language"), key="w2v_word")
    vector = get_word_vector(model, word)
    st.metric("Dense vector dimensions", len(vector))
    norm = cosine_similarity(vector, vector, explain=True)["magnitude_a"]
    st.metric("Vector norm", round(norm, 6))
    st.table([{"Dimension": i, "Value": value} for i, value in enumerate(vector[:10])])
    with st.expander("Full dense vector"):
        st.json(vector)
    one_hot = one_hot_encode(word, vocabulary)
    st.table([{"Representation": "One-hot", "Dimensions": len(one_hot), "Nonzero coordinates": sum(one_hot)}, {"Representation": "Dense Word2Vec", "Dimensions": len(vector), "Nonzero coordinates": sum(value != 0 for value in vector)}])
    st.caption(f"One-hot active index: {vocabulary.index(word)}. Dense values are learned, not hand-assigned semantic labels.")
    with st.expander("Full one-hot vector"):
        st.json(one_hot)
    st.subheader("5. Compare two words: same cosine, new representation")
    left, right = st.columns(2)
    with left:
        word_a = st.selectbox("Word A", vocabulary, index=vocabulary.index("language"), key="w2v_word_a")
    with right:
        word_b = st.selectbox("Word B", vocabulary, index=vocabulary.index("text"), key="w2v_word_b")
    manual = calculate_word_similarity(model, word_a, word_b)
    gensim_score = float(model.wv.similarity(word_a, word_b))
    st.table([{"Pair": f"{word_a} / {word_b}", "Manual cosine": manual, "Gensim similarity": gensim_score, "Absolute difference": abs(manual - gensim_score)}])
    st.caption("Try language/text, machine/learning, machine/database, student/course and attendance/examination. No pair is guaranteed to be most similar.")
    st.subheader("6. Similar words")
    top_n = st.slider("Nearest words", 1, 10, 5, key="w2v_top_n")
    neighbours = find_similar_words(model, word, top_n)
    st.table([{"Rank": rank, "Word": term, "Cosine": score} for rank, (term, score) in enumerate(neighbours, 1)])
    st.write("Nearest means close in this model's learned space. Association is not necessarily synonymy, interchangeability or a factual relationship.")
    with st.expander("Out-of-vocabulary experiment"):
        probe = st.text_input("Word to retrieve", value="zzworkshopunknown", key="w2v_oov")
        try:
            st.json(get_word_vector(model, probe))
        except ValueError as error:
            st.info(str(error))
    st.subheader("7. Compare CBOW and Skip-Gram")
    if st.checkbox("Show both architectures with otherwise identical settings", key="w2v_compare"):
        other = cached_model(tokens, vector_size, window, 1 - sg)
        cbow, skipgram = (model, other) if sg == 0 else (other, model)
        col_a, col_b = st.columns(2)
        for column, label, candidate in [(col_a, "CBOW", cbow), (col_b, "Skip-Gram", skipgram)]:
            with column:
                st.write(label + " — " + word)
                st.table([{"Word": term, "Cosine": score} for term, score in find_similar_words(candidate, word, top_n)])
        st.caption("Prediction direction can change geometry; neither architecture is universally superior. Raw vector coordinates from separately trained models are not aligned axes.")
    st.subheader("8. Optional PCA projection")
    if st.checkbox("Show 2D PCA projection", key="w2v_pca"):
        suggested = [term for term in ["language", "text", "nlp", "machine", "learning", "deep", "model", "data", "student", "course", "attendance", "examination", "network", "database"] if term in model.wv]
        selected = st.multiselect("Words to project (2–20)", vocabulary, default=suggested, max_selections=20, key="w2v_project_words")
        st.warning("2D projection for visualization only. PCA fits the selected subset and discards dimensions; apparent 2D closeness can distort original-space similarity.")
        if len(selected) >= 2:
            points = project_word_vectors(model, selected)
            st.vega_lite_chart(spec={"data": {"values": points}, "layer": [
                {"mark": {"type": "point", "filled": True, "size": 90}, "encoding": {"x": {"field": "x", "type": "quantitative", "title": "PCA component 1"}, "y": {"field": "y", "type": "quantitative", "title": "PCA component 2"}, "tooltip": [{"field": "word"}]}},
                {"mark": {"type": "text", "dy": -10}, "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"}, "text": {"field": "word"}}}
            ]})
            st.table(points)
        else:
            st.info("Select at least two distinct words to project.")
    st.subheader("9. GloVe and FastText: concepts only")
    st.write("Word2Vec learns through prediction in local contexts. GloVe learns from global word co-occurrence statistics. FastText uses character subword information: learn, learning and learner share pieces, which can help rare words, morphological variation and unseen forms. We do not train/download GloVe or FastText models here.")
    st.subheader("10. Static-vector limitations")
    st.code("bank approved the loan\nstudents sat on the river bank", language="text")
    st.write("Traditional Word2Vec: one word → one vector. If bank is in the vocabulary, both sentences retrieve the same stored bank vector; this model does not choose a different sense from context. The workshop corpus need not contain bank to explain this architectural limitation.")
    st.success("Exercise 5 milestone: words have learned vector representations. Next, represent complete sentences.")
    st.info("Word2Vec represents individual words. How do we represent an entire sentence as one meaningful vector? Sentence embeddings come next; no sentence/document embeddings or semantic search are implemented here.")

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
