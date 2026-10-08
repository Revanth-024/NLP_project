"""Generated student UI: infrastructure retained, expected TODOs explained."""
import streamlit as st
try:
    """Exercises 8–9. UI/caching are starter code; student math lives in src."""
    import streamlit as st
    from src.transformers_nlp import (attention_example, analyze_sentiment, recognize_entities,
                                      answer_from_context)
    from src.transformer_resources import MODELS, load_transformer_pipeline, TransformerModelUnavailable

    st.set_page_config(page_title="Transformer Lab", page_icon="📚", layout="wide")
    st.title("Attention, Transformers, BERT and GPT")
    st.caption("Exercise 8 — manual attention · Exercise 9 — instructor-guided pretrained NLP")
    st.header("1 — Why context matters")
    st.write('Compare “bank approved the loan” with “students sat on the river bank”. '
             'Traditional Word2Vec gives bank one static vector. Contextual token representations '
             'can vary with surrounding words; they do not guarantee perfect interpretation.')

    st.header("2 — Query, Key and Value")
    st.table([{"Role": "Query", "Intuition": "What information am I looking for?"},
              {"Role": "Key", "Intuition": "What information does each token advertise?"},
              {"Role": "Value", "Intuition": "What information can each token contribute?"}])
    st.write("Query matches keys → scores → softmax weights → weighted values → context.")

    st.header("3–4 — Manual attention calculator")
    st.info("Simplified educational attention example. These user-entered vectors and weights "
            "are an analogy, not actual learned BERT/GPT attention.")
    st.latex(r"s_i=Q\cdot K_i,\quad w_i=\frac{e^{s_i}}{\sum_j e^{s_j}},\quad c=\sum_i w_i V_i")
    left, right = st.columns(2)
    with left:
        qx = st.number_input("Query dimension 1", value=1.0, min_value=-10.0, max_value=10.0, key="attention_qx")
    with right:
        qy = st.number_input("Query dimension 2", value=0.0, min_value=-10.0, max_value=10.0, key="attention_qy")
    left, right = st.columns(2)
    with left:
        st.subheader("Keys: three vectors")
        keys = st.data_editor([{"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}, {"x": 0.5, "y": 0.5}],
                              hide_index=True, key="attention_keys")
    with right:
        st.subheader("Values: three vectors")
        values = st.data_editor([{"x": 2.0, "y": 0.0}, {"x": 0.0, "y": 2.0}, {"x": 1.0, "y": 1.0}],
                                hide_index=True, key="attention_values")
    scaled = st.checkbox("Instructor comparison: divide scores by √dₖ (not an additional TODO)", key="attention_scaled")
    try:
        key_vectors = [[row["x"], row["y"]] for row in keys]
        value_vectors = [[row["x"], row["y"]] for row in values]
        result = attention_example([qx, qy], key_vectors, value_vectors, scaled=scaled)
        st.write("Q =", [qx, qy])
        rows = [{"Key": f"K{i+1}", "Dot score": float(score), "Effective score": float(effective),
                 "Softmax weight": float(weight), "Weighted V₁": float(value[0]), "Weighted V₂": float(value[1])}
                for i, (score, effective, weight, value) in enumerate(zip(result["scores"], result["effective_scores"],
                                                                         result["weights"], result["weighted_values"]))]
        st.dataframe(rows, hide_index=True)
        st.bar_chart(rows, x="Key", y="Softmax weight")
        st.write("Weights sum:", float(result["weights"].sum()))
        st.write("Final context vector:", result["context"].tolist())
    except (ValueError, TypeError) as error:
        st.warning(f"Check the numeric vectors: {error}")
    st.write("Softmax subtracts the maximum score before exponentiation to avoid overflow. "
             "This preserves the ratios. Mathematical weights are positive; extreme gaps can "
             "round tiny computed weights to zero.")
    st.write('Language analogy: “The student submitted the assignment because it was due.” '
             'A representation of it can use context from assignment. Our invented vectors '
             'illustrate selection and combination, not a measured coreference model.')

    st.header("5 — Self-attention, scaling and multiple heads")
    st.write("In self-attention, Q, K and V are derived from tokens in the same sequence using "
             "learned projections W_Q, W_K and W_V. Each row in an attention matrix corresponds "
             "to a token asking for information; columns identify tokens attended to.")
    st.code("             attends to\n             T1  T2  T3  T4\ntoken T1     ... ... ... ...\ntoken T2     ... ... ... ...\ntoken T3     ... ... ... ...\ntoken T4     ... ... ... ...", language=None)
    st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^{\mathsf T}/\sqrt{d_k})V")
    st.write("Larger dimensions can produce large dot products; √dₖ scaling helps prevent "
             "overly concentrated softmax weights. Multiple heads use different learned "
             "projections and can capture different relationships. A head has no guaranteed "
             "human-readable role; attention weights are not explanations of model reasoning.")

    st.header("6 — Transformer architecture")
    st.write("Tokenizer → token embeddings + positional information → stacked Transformer blocks "
             "(self-attention, feed-forward layers, residual connections, normalization) → task output. "
             "Attention mixes information across tokens; feed-forward layers transform each position. "
             "Residual paths and normalization help stacked blocks train and operate reliably.")
    st.header("7 — BERT")
    st.write("Bidirectional Encoder Representations from Transformers: an encoder-oriented model "
             "building contextual representations from surrounding tokens. Masked language modeling "
             "asks it to predict hidden tokens. Fine-tuned encoder models commonly support "
             "classification, NER and extractive QA.")
    st.code("The student submitted the [MASK] before the deadline.", language=None)
    st.caption("Conceptual masked-token example; no masked-language-model download or inference here.")
    st.header("8 — GPT")
    st.write("Generative Pre-trained Transformer: typically decoder-oriented with causal attention. "
             "Autoregressive language modeling predicts the next token from preceding tokens, "
             "then repeats the operation to generate text.")
    st.code("Natural language processing helps computers ... → next token → next token", language=None)
    st.caption("GPT is taught conceptually. No generative model is loaded.")
    st.header("9 — BERT vs GPT and yesterday’s encoder")
    st.table([{"Feature": "Typical architecture", "BERT": "Encoder", "GPT": "Decoder"},
              {"Feature": "Context style", "BERT": "Bidirectional", "GPT": "Causal/autoregressive"},
              {"Feature": "Pretraining intuition", "BERT": "Masked tokens", "GPT": "Next token"},
              {"Feature": "Common strength", "BERT": "Understanding/representation", "GPT": "Generation"},
              {"Feature": "Example uses", "BERT": "Classification, NER, extractive QA", "GPT": "Generation, assistants"}])
    st.write("These are useful typical distinctions, not absolute boundaries. Sentence Transformers "
             "such as all-MiniLM-L6-v2 use Transformer encoders and similarity-oriented training to "
             "produce sentence vectors. You already used Transformer technology in Tasks 08–09.")

    @st.cache_resource(max_entries=3)
    def cached_task_pipeline(task):
        return load_transformer_pipeline(task)


    st.header("10 — Pretrained Transformer NLP")
    st.write("Choose one task and run it. Model loading, tokenization, inference and caching are "
             "starter code; Exercise 9 has no algorithm TODO. Models run on CPU from a preloaded "
             "local cache. No download occurs on this page.")
    task = st.selectbox("Task", ["sentiment", "ner", "qa"], key="transformer_task")
    st.caption(f"Model: {MODELS[task]['id']} · Apache-2.0 · pinned revision {MODELS[task]['revision']}")
    if task == "sentiment":
        text = st.text_area("Sentiment text", "The NLP workshop was engaging and easy to follow.", key="sentiment_text")
    elif task == "ner":
        text = st.text_area("Entity text", "Riya studies Natural Language Processing at Hindu College of Engineering.", key="ner_text")
    else:
        context = st.text_area("Supplied context", "Natural Language Processing is offered in Semester 6. "
                               "The course introduces text preprocessing, TF-IDF, word embeddings, attention, "
                               "Transformers and semantic retrieval.", key="qa_context")
        question = st.text_input("Question", "In which semester is Natural Language Processing offered?", key="qa_question")
        st.info("Extractive QA selects an answer span from the context you supply. This is not RAG. "
                "This SQuAD-v1 model can return an incorrect span for an unsupported question; "
                "it does not reliably abstain.")
    if st.button("Analyze selected Transformer task", key="transformer_run"):
        try:
            with st.spinner("Loading cached CPU model and running inference…"):
                pipe = cached_task_pipeline(task)
                if task == "sentiment":
                    st.table([analyze_sentiment(pipe, text)])
                elif task == "ner":
                    entities = recognize_entities(pipe, text)
                    if entities:
                        st.dataframe(entities, hide_index=True)
                    else:
                        st.info("No entities recognized. Try another sentence; absence is also an observation.")
                else:
                    answer = answer_from_context(pipe, question, context)
                    st.table([answer])
                    st.write("Exact context slice:", context[answer["start"]:answer["end"]])
        except (TransformerModelUnavailable, ValueError, RuntimeError) as error:
            st.warning(str(error))
            st.code(f"python -m src.transformer_resources --download --task {task}\npython -m src.transformer_resources --task {task}", language="bash")
    st.caption("Task output scores are model-specific estimates, not universal truth or factual confidence. "
               "NER labels are learned categories; recognized entities are not verified university facts. "
               "Short input only: at most 400 tokens; QA question at most 64.")
    st.header("11 — What models still cannot guarantee")
    st.write("Knowledge can be incomplete or outdated. Generated text can be unsupported. "
             "A pretrained model does not automatically know our private fictional documents. "
             "Task models need suitable input/context, have length limits and can misclassify text.")
    st.header("12 — Bridge toward document processing and RAG")
    st.write("What if we do not know which document contains the context? Retrieval can find "
             "evidence; later a language model can use that evidence. No retrieval-to-QA or "
             "generation pipeline is implemented here. Task 09 exposed long-document truncation; "
             "next we must extract documents and divide them into meaningful units.")
    st.caption("Overall Checkpoint5 — Transformer experiments work. The next labs build document retrieval and RAG.")

except NotImplementedError as error:
    st.info(str(error))
    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")
