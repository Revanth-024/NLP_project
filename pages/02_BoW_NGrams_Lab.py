"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Starter UI for Exercise 2; shared src functions compute every representation."""
    import json
    from pathlib import Path
    import streamlit as st
    from src.preprocessing import lowercase_text, remove_punctuation, tokenize_words
    from src.classical_nlp import (
        build_vocabulary, create_vocabulary_index, one_hot_encode, bag_of_words,
        build_document_term_matrix, generate_ngrams, countvectorizer_matrix,
    )

    st.set_page_config(page_title="BoW and N-Grams Lab", page_icon="📚")
    st.title("Exercise 2 — One-Hot Encoding, Bag of Words and N-Grams")
    st.caption("Student starter state. Student cores 2.1–2.4 live in src/classical_nlp.py; infrastructure is provided.")
    st.write("Input → Preprocessed Tokens → Vocabulary → Numerical Representation → Matrix → Interpretation")
    root = Path(__file__).resolve().parents[1]
    collections = json.loads((root / "data/examples/bow_examples.json").read_text(encoding="utf-8"))["collections"]
    examples = {collection["id"]: collection["texts"] for collection in collections}
    st.subheader("1. Enter documents")
    example = st.selectbox("Example collection", list(examples), key="bow_example")
    if "bow_documents" not in st.session_state:
        st.session_state["bow_documents"] = "\n".join(examples["BOW-01"])
    if st.button("Load collection"):
        st.session_state["bow_documents"] = "\n".join(examples[example])
    raw = st.text_area("One document per line (2–5 non-empty lines)", key="bow_documents", height=150, max_chars=5000)
    if len(raw) > 5000:
        st.warning("Use up to 5,000 characters for a small inspectable corpus."); st.stop()
    texts = [line.strip() for line in raw.splitlines() if line.strip()]
    if not 2 <= len(texts) <= 5:
        st.info("Enter 2–5 non-empty document lines to inspect the shared representation.")
        st.stop()
    documents = [tokenize_words(remove_punctuation(lowercase_text(text))) for text in texts]
    labels = [f"D{i + 1}" for i in range(len(documents))]
    st.subheader("2. Preprocessed documents")
    st.caption("Exercise 1 lowercase → punctuation spaces → tokenization. Stopwords are retained; no stemming or lemmatization is applied.")
    for label, text, tokens in zip(labels, texts, documents):
        st.write(f"{label}: {text}")
        st.json(tokens)
    vocabulary = build_vocabulary(documents)
    index = create_vocabulary_index(vocabulary)
    st.subheader("3. Vocabulary and coordinate index")
    st.json(vocabulary)
    st.metric("Vocabulary size", len(vocabulary))
    if not vocabulary:
        st.info("These documents contain no tokens after punctuation handling. Enter words to build numerical representations.")
        st.stop()
    st.table([{"Term": word, "Index": position} for word, position in index.items()])
    st.subheader("4. One-hot encoding")
    word = st.selectbox("Vocabulary word", vocabulary, key="bow_word")
    vector = one_hot_encode(word, vocabulary)
    st.dataframe([{"Term": term, "Index": i, "One-hot": value} for i, (term, value) in enumerate(zip(vocabulary, vector))], hide_index=True)
    st.caption("Length = vocabulary size. Exactly one coordinate is active. Most coordinates are zero; these vectors contain no semantic closeness.")
    if len(vocabulary) <= 20:
        with st.expander("Complete one-hot table"):
            st.table([{"Word": term, **dict(zip(vocabulary, one_hot_encode(term, vocabulary)))} for term in vocabulary])
    else:
        st.caption("The complete one-hot table is omitted above 20 terms to keep the lab readable.")
    st.subheader("5. Manual Bag of Words")
    for label, tokens in zip(labels, documents):
        st.write(f"{label}: {bag_of_words(tokens, vocabulary)} — columns follow the vocabulary above.")
    matrix = build_document_term_matrix(documents, vocabulary)
    st.subheader("6. Document-term matrix")
    st.dataframe([{"Document": label, **dict(zip(vocabulary, row))} for label, row in zip(labels, matrix)], hide_index=True)
    st.caption(f"Shape: {len(documents)} documents × {len(vocabulary)} terms. Each row sum equals that document's token count.")
    st.subheader("7. N-Grams")
    n = st.selectbox("Window size n", [1, 2, 3], key="bow_n")
    for label, tokens in zip(labels, documents):
        st.write(label)
        st.json([list(gram) for gram in generate_ngrams(tokens, n)])
    st.caption("Contiguous windows retain limited local order. Repeated windows remain repeated; windows never cross documents.")
    st.subheader("8. Manual vs CountVectorizer")
    library = countvectorizer_matrix(documents, vocabulary)
    st.write("Library vocabulary (fixed to the same coordinate index):")
    st.json(index)
    st.dataframe([{"Document": label, **dict(zip(vocabulary, row))} for label, row in zip(labels, library)], hide_index=True)
    if matrix == library:
        st.success("Manual and CountVectorizer matrices match exactly, including column order.")
    else:
        st.error("Matrices differ. Check the manual TODO cores and coordinate alignment.")
    st.caption("analyzer=list consumes the already-tokenized documents; vocabulary is fixed, lowercase=False and token_pattern=None. Raw-text defaults otherwise lowercase and exclude single-character tokens. The callable analyzer also bypasses ngram_range; this comparison is for unigram counts.")
    st.subheader("9. Limitations")
    order_tokens = [tokenize_words(remove_punctuation(lowercase_text(text))) for text in ["Dog bites man.", "Man bites dog."]]
    order_vocab = build_vocabulary(order_tokens)
    st.table([{"Document": text, **dict(zip(order_vocab, row))} for text, row in zip(["Dog bites man.", "Man bites dog."], build_document_term_matrix(order_tokens, order_vocab))])
    st.write("Different meanings, identical unigram counts. Compare their bigrams:")
    for tokens in order_tokens:
        st.json([list(gram) for gram in generate_ngrams(tokens, 2)])
    st.table([{"Word": term, **dict(zip(["automobile", "car"], one_hot_encode(term, ["automobile", "car"])))} for term in ["car", "automobile"]])
    st.write("Car and automobile occupy separate coordinates despite related meanings. Embeddings will address this later. Increasing the N-Gram range increases possible features and sparsity.")
    common = [["the", "the", "the", "language"], ["the", "data"]]
    common_vocab = build_vocabulary(common)
    st.table([{"Document": f"C{i + 1}", **dict(zip(common_vocab, row))} for i, row in enumerate(build_document_term_matrix(common, common_vocab))])
    st.write("Every occurrence contributes equally: repeated common terms can dominate raw counts.")
    st.info("We can now represent documents numerically, but every occurrence contributes equally. Common words may dominate the representation. How can we measure which words are actually informative? Next: TF-IDF from scratch.")

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
