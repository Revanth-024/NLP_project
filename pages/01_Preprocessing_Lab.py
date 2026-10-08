"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Starter-provided UI for Exercise 1; transformations live in src."""
    import json
    from pathlib import Path

    import streamlit as st

    from src.nltk_resources import NLTKResourceError
    from src.preprocessing import (
        build_vocabulary,
        calculate_word_frequencies,
        lemmatize_words,
        lowercase_text,
        remove_punctuation,
        remove_stopwords,
        stem_words,
        tag_parts_of_speech,
        tokenize_words,
    )

    st.set_page_config(page_title="Preprocessing Lab", page_icon="📚")
    st.title("Exercise 1 — Text Preprocessing")
    st.write("Input → Intermediate Representation → Result")
    st.caption("Student starter state. Student boundaries 1.1–1.7 are in src/preprocessing.py; UI and setup are provided.")
    st.write("Inspect each operation separately before building numerical representations. Changing the preprocessing policy changes what the representation retains.")

    default_text = (
        "Natural Language Processing is amazing! Students are learning how computers "
        "process human languages, and the students are building useful NLP applications."
    )
    root = Path(__file__).resolve().parents[1]
    examples = {"Workshop default": default_text}
    example_data = json.loads((root / "data/examples/preprocessing_examples.json").read_text(encoding="utf-8"))
    examples.update({item["id"]: item["text"] for item in example_data["examples"]})

    example_name = st.selectbox("Dataset example", list(examples), key="example_name")
    if "raw_text" not in st.session_state:
        st.session_state["raw_text"] = default_text
    if st.button("Use selected example"):
        st.session_state["raw_text"] = examples[example_name]
    raw_text = st.text_area("Input text", key="raw_text", height=130, max_chars=5000)
    if len(raw_text) > 5000:
        st.warning("Use up to 5,000 characters in this interactive lab; split longer text into smaller experiments."); st.stop()

    st.subheader("Pipeline choices")
    left, middle, right = st.columns(3)
    with left:
        use_lowercase = st.checkbox("Lowercase", value=True, key="use_lowercase")
    with middle:
        handle_punctuation = st.checkbox("Replace punctuation with spaces", value=True, key="handle_punctuation")
    with right:
        filter_stopwords = st.checkbox("Remove English stopwords", value=True, key="filter_stopwords")
    st.warning("Stopword removal is a task-dependent choice, not an automatic requirement for every NLP system. English stopwords include negation such as 'not'.")

    st.subheader("1. Original text")
    st.code(raw_text or "(empty input)", language="text")

    lowercase = lowercase_text(raw_text) if use_lowercase else raw_text
    st.subheader("2. Lowercase text")
    st.caption("1.1: normalize case only." if use_lowercase else "Lowercasing is bypassed; original case is retained.")
    st.code(lowercase or "(empty input)", language="text")

    punctuation_handled = remove_punctuation(lowercase) if handle_punctuation else lowercase
    st.subheader("3. Punctuation-handled text")
    st.caption("1.2: punctuation becomes spaces so neighbouring words do not merge. Case and whitespace are not otherwise changed." if handle_punctuation else "Punctuation handling is bypassed. The tokenizer can emit punctuation tokens.")
    st.code(punctuation_handled or "(empty input)", language="text")
    st.caption("This simple policy splits hyphens, apostrophes and decimal points. Punctuation can carry meaning; inspect the tradeoff.")

    tokens = tokenize_words(punctuation_handled)
    st.subheader("4. Tokens")
    st.caption("1.3: NLTK WordPunct tokenization preserves order and repeated tokens; no tokenizer model download is needed.")
    st.json(tokens)

    st.subheader("5. Tokens without stopwords")
    filtered_tokens = tokens
    filter_available = True
    if filter_stopwords:
        try:
            filtered_tokens = remove_stopwords(tokens)
            st.caption("1.4: filter against the provided English set, without modifying the input list.")
            st.json(filtered_tokens)
        except NLTKResourceError as error:
            filter_available = False
            st.warning(str(error))
            st.caption("Stopword filtering is unavailable. Later displays use the original tokens, not a simulated filtered result.")
    else:
        st.caption("Stopword filtering is bypassed; all tokens are retained.")
        st.json(filtered_tokens)

    st.info("Stemming and lemmatization below are parallel alternatives applied to the same stage-5 tokens. We do not lemmatize already-stemmed words.")
    st.subheader("6. Stemmed tokens")
    stemmed = stem_words(filtered_tokens)
    st.caption("1.5: Porter stems can be non-dictionary forms, for example studies → studi. Porter also lowercases tokens by default, even when the earlier lowercase stage is bypassed.")
    st.json(stemmed)

    st.subheader("7. Lemmatized tokens")
    pos_options = {"Noun (n, default)": "n", "Verb (v)": "v", "Adjective (a)": "a", "Adverb (r)": "r"}
    pos_label = st.selectbox("Assumed WordNet POS", list(pos_options), key="lemma_pos")
    assumed_pos = pos_options[pos_label]
    st.caption("1.6: this assumes the selected POS for every token; it is not automatic grammatical analysis. A noun default may leave learning unchanged, while a verb assumption can produce learn.")
    lemmas = None
    try:
        lemmas = lemmatize_words(filtered_tokens, pos=assumed_pos)
        st.json(lemmas)
    except NLTKResourceError as error:
        st.warning(str(error))
        st.caption("Lemmatization is unavailable. Other operations remain usable; no fake lemmas are shown.")

    st.subheader("Stemming versus lemmatization")
    st.write("Stemming applies heuristic rules and can be aggressive. Lemmatization aims for dictionary forms and depends on linguistic assumptions and resources. Neither is universally better.")
    if lemmas is not None:
        comparison = {}
        for token, stem, lemma in zip(filtered_tokens, stemmed, lemmas):
            comparison.setdefault(token, {"Original": token, "Stem": stem, f"Lemma ({assumed_pos})": lemma})
        if comparison:
            st.table(list(comparison.values()))
        else:
            st.caption("Enter text to compare forms.")
    else:
        st.caption("Install WordNet to enable the lemma comparison.")

    st.subheader("8. Vocabulary")
    base_label = "Filtered tokens" if filter_available and filter_stopwords else "Tokens (unfiltered)"
    representations = {base_label: filtered_tokens, "Stemmed tokens": stemmed}
    if lemmas is not None:
        representations["Lemmatized tokens"] = lemmas
    representation = st.radio("Tokens used for vocabulary and frequencies", list(representations), index=len(representations) - 1, key="representation")
    selected_tokens = representations[representation]
    vocabulary = build_vocabulary(selected_tokens)
    st.caption("1.7: vocabulary is the sorted unique token set of the selected representation. Flatten tokenized documents first when working with a corpus.")
    st.json(vocabulary)
    st.metric("Vocabulary size", len(vocabulary))

    st.subheader("9. Word frequencies")
    frequencies = calculate_word_frequencies(selected_tokens)
    st.caption("1.7: repetitions contribute to counts. The sum of frequencies equals the number of selected tokens; it need not equal vocabulary size.")
    if frequencies:
        rows = [{"Token": token, "Frequency": count} for token, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))]
        st.table(rows)
    else:
        st.caption("No tokens to count.")
    st.metric("Total selected tokens", sum(frequencies.values()))

    with st.expander("Instructor demo: parts of speech (optional)"):
        st.caption("Starter code calls an English POS tagger. Students do not implement it. Predicted tags can be wrong, especially for invented names; this is not NER.")
        demo_text = st.text_input("POS demo sentence", value="Dr. Nivora Pellin teaches Natural Language Processing at Hindu College of Engineering.", key="pos_demo_text")
        if st.checkbox("Run optional POS demonstration", key="show_pos"):
            try:
                tags = tag_parts_of_speech(tokenize_words(demo_text))
                st.table([{"Token": token, "Predicted POS": tag} for token, tag in tags])
                st.caption("NN/NNP: noun/proper noun; VB variants: verb; JJ: adjective; IN: preposition; DT: determiner. A POS label is not a named-entity label.")
            except NLTKResourceError as error:
                st.info(str(error))
        st.write("For Named Entity Recognition, open the existing Transformer Lab. No spaCy model is required.")

    st.success("Checkpoint 1: inspect the stages, compare forms, and run the preprocessing tests. The reference functions are complete; a student starter checkpoint will be prepared later.")

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
