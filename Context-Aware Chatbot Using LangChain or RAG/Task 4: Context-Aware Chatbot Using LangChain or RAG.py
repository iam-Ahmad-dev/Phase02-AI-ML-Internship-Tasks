"""
Task 4: Context-Aware Chatbot Using LangChain + RAG
====================================================
Architecture:
  - Knowledge base : Built-in AI/ML corpus (no download needed)
  - Embeddings     : HuggingFace all-MiniLM-L6-v2 (FREE, runs locally)
  - Vector store   : FAISS (in-memory)
  - LLM            : HuggingFace Inference API (Mistral-7B, FREE tier)
  - Memory         : Manual chat history via st.session_state (no deprecated langchain.memory)
  - UI             : Streamlit

Run:
  streamlit run chatbot_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Modern LangChain imports — compatible with langchain-core >= 0.3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# KNOWLEDGE BASE CORPUS
# ─────────────────────────────────────────────
CORPUS = [
    """
    Machine Learning Overview
    Machine learning (ML) is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on developing
    computer programs that can access data and use it to learn for themselves.

    The three main types of machine learning are:
    1. Supervised Learning: The algorithm learns from labeled training data. Examples include
       linear regression, logistic regression, decision trees, and neural networks.
    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Examples include
       k-means clustering, hierarchical clustering, and PCA (Principal Component Analysis).
    3. Reinforcement Learning: An agent learns by interacting with an environment and receiving
       rewards or penalties. Used in robotics, game playing (AlphaGo), and autonomous vehicles.
    """,
    """
    Deep Learning and Neural Networks
    Deep learning is a subset of machine learning that uses neural networks with many layers
    (deep neural networks) to learn representations of data. It has revolutionized fields like
    computer vision, natural language processing, and speech recognition.

    Key deep learning architectures:
    - Convolutional Neural Networks (CNNs): Excellent for image recognition and computer vision tasks.
    - Recurrent Neural Networks (RNNs): Designed for sequential data like text and time series.
      LSTMs (Long Short-Term Memory) and GRUs solve the vanishing gradient problem.
    - Transformers: The dominant architecture for NLP. Based on the self-attention mechanism.
      Models like BERT, GPT, and T5 are all transformer-based.
    - GANs: Two networks (generator and discriminator) compete to produce realistic synthetic data.
    """,
    """
    Natural Language Processing (NLP)
    NLP is a branch of AI that deals with the interaction between computers and human language.

    Key NLP tasks:
    - Text Classification: Categorizing text into predefined classes (e.g., spam detection, sentiment analysis).
    - Named Entity Recognition (NER): Identifying persons, organizations, and locations in text.
    - Machine Translation: Automatically translating text between languages.
    - Question Answering: Extracting answers from a context given a natural language question.
    - Text Summarization: Condensing long documents into shorter summaries.
    - Sentiment Analysis: Determining the emotional tone of text (positive, negative, neutral).

    Modern NLP relies on pretrained language models like BERT, GPT-4, and LLaMA.
    """,
    """
    Large Language Models (LLMs) and BERT
    Large Language Models are neural networks trained on vast amounts of text data.

    BERT (Bidirectional Encoder Representations from Transformers):
    - Developed by Google in 2018. Uses a bidirectional transformer encoder.
    - Pre-trained on masked language modeling and next sentence prediction.
    - Fine-tuned for downstream tasks like text classification, NER, and question answering.
    - bert-base-uncased has 110 million parameters and 12 transformer layers.
    - AG News is a common benchmark with 120,000 samples across 4 categories: World, Sports, Business, Sci/Tech.

    GPT (Generative Pre-trained Transformer):
    - Developed by OpenAI. Uses a unidirectional transformer decoder.
    - GPT-4 is multimodal — understands both text and images.
    """,
    """
    Retrieval-Augmented Generation (RAG)
    RAG combines information retrieval with text generation to reduce hallucinations.

    RAG Pipeline:
    1. Indexing: Documents are split into chunks, embedded into vectors, stored in a vector DB.
    2. Retrieval: Query is embedded and top-k most similar chunks are retrieved via ANN search.
    3. Generation: Retrieved chunks + query are passed to the LLM as context for answer generation.

    Benefits of RAG:
    - Reduces hallucinations by grounding answers in retrieved facts.
    - Allows LLMs to access information beyond their training cutoff.
    - More efficient than fine-tuning for adding domain-specific knowledge.
    - Enables source attribution.
    """,
    """
    Scikit-learn and ML Pipelines
    Scikit-learn is the most popular Python library for classical machine learning.

    Key components:
    - Pipeline API: Chains preprocessing and models into one object, preventing data leakage.
    - ColumnTransformer: Applies different preprocessing to numeric and categorical columns.
    - GridSearchCV: Exhaustive hyperparameter tuning with cross-validation.
    - StandardScaler: Standardizes features to zero mean and unit variance.
    - OneHotEncoder: Converts categorical variables into binary columns.
    - drop='first' in OneHotEncoder avoids the dummy variable trap (multicollinearity).

    Common classifiers: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM.
    joblib is used to export trained pipelines for production reuse.
    """,
    """
    Model Evaluation Metrics
    Classification metrics:
    - Accuracy: Fraction of correct predictions. Misleading for imbalanced datasets.
    - Precision: TP / (TP + FP). Of predicted positives, how many are truly positive?
    - Recall: TP / (TP + FN). Of actual positives, how many did we find?
    - F1-Score: Harmonic mean of precision and recall. Best for imbalanced data.
    - ROC-AUC: Measures model's discrimination ability across all thresholds.
    - Confusion Matrix: Table of TP, TN, FP, FN counts.

    Regression metrics:
    - MAE (Mean Absolute Error): Average absolute difference between predictions and actuals.
    - RMSE (Root Mean Squared Error): Penalizes large errors more than MAE.
    - R-squared: Proportion of variance explained by the model.

    Cross-validation: k-Fold and Stratified k-Fold for reliable generalization estimates.
    """,
    """
    Transfer Learning and Fine-tuning
    Transfer learning reuses a model trained on one task as the starting point for another.

    How it works:
    1. Pretrain on a large dataset (e.g., BERT on Wikipedia + BooksCorpus).
    2. Fine-tune on a smaller, task-specific dataset.
    3. The model retains general knowledge and adapts to the new task.

    Examples:
    - BERT fine-tuned on AG News for 4-class news classification (World/Sports/Business/Sci-Tech).
    - ResNet pretrained on ImageNet, fine-tuned for medical imaging.

    Parameter-efficient fine-tuning (PEFT) methods like LoRA allow fine-tuning large models
    by updating only a small fraction of parameters, making it feasible on consumer hardware.
    """,
    """
    Customer Churn Prediction
    Customer churn refers to when customers stop doing business with a company.

    Telco Churn Dataset:
    - 7,043 customers, 20 features, ~27% churn rate (imbalanced).
    - Key predictors: Contract type, tenure, MonthlyCharges, InternetService.
    - Use F1-score and ROC-AUC rather than accuracy due to class imbalance.

    ML approach:
    - Logistic Regression and Random Forest are strong baselines for tabular churn data.
    - GridSearchCV tunes hyperparameters: C for LogReg, n_estimators/max_depth for RF.
    - Both models can be searched in one GridSearchCV using a list-style param_grid.
    - Export the trained pipeline with joblib for production deployment.
    """,
    """
    Vector Databases and Embeddings
    Vector databases store and search high-dimensional embeddings efficiently.

    Popular vector databases:
    - FAISS (Facebook AI Similarity Search): Open-source, in-memory, extremely fast.
    - Pinecone: Managed cloud vector DB, scales to billions of vectors.
    - Chroma: Lightweight, open-source, easy to use with LangChain.

    Embedding models:
    - all-MiniLM-L6-v2: Free, lightweight (80MB), runs locally, 384 dimensions.
    - all-mpnet-base-v2: Higher quality than MiniLM, larger model.
    - OpenAI text-embedding-ada-002: High quality, paid API.

    Similarity search methods:
    - Cosine similarity: Measures angle between vectors. Most common.
    - L2 (Euclidean) distance: Measures geometric distance.
    """
]

RAG_PROMPT_TEMPLATE = """You are a helpful AI/ML assistant. Use the retrieved context below
to answer the question. If the context does not contain enough information, say so clearly.
Keep your answer concise and factual.

Conversation history:
{chat_history}

Retrieved context:
{context}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base (one-time setup ~30s)...")
def build_vectorstore():
    """
    Split corpus into chunks, embed locally with HuggingFace (free, no API needed),
    and index in FAISS. Cached so it runs only once per session.
    """
    documents = [Document(page_content=text) for text in CORPUS]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def call_llm(prompt: str, hf_token: str) -> str:
    """
    Call LLM via HuggingFace InferenceClient using chat_completion (conversational) API.
    Uses mistralai/Mistral-7B-Instruct-v0.2 with the messages format.
    """
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=hf_token,
    )
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def answer_question(question: str, vectorstore, hf_token: str, chat_history: list) -> tuple:
    """
    RAG pipeline:
    1. Retrieve top-3 relevant chunks from FAISS
    2. Format chat history as string (context memory)
    3. Build prompt and call LLM via InferenceClient
    Returns (answer, source_chunks)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content.strip() for doc in docs])

    history_text = ""
    for turn in chat_history[-6:]:
        history_text += f"Human: {turn['human']}\nAssistant: {turn['ai']}\n\n"

    prompt = RAG_PROMPT_TEMPLATE.format(
        chat_history=history_text if history_text else "None yet.",
        context=context,
        question=question,
    )

    answer = call_llm(prompt, hf_token)

    sources = [doc.page_content.strip() for doc in docs]
    return answer, sources


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    load_dotenv()

    st.set_page_config(
        page_title="RAG Chatbot — AI/ML Knowledge Base",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Context-Aware RAG Chatbot")
    st.caption("LangChain · FAISS · HuggingFace · Streamlit  |  No OpenAI key needed")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown(
            "Get your **free** HuggingFace token at "
            "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)\n\n"
            "Choose **Read** access when creating the token."
        )

        hf_token = st.text_input(
            "HuggingFace API Token",
            type="password",
            value=os.getenv("HUGGINGFACE_API_TOKEN", ""),
            placeholder="hf_...",
        )

        st.divider()
        st.markdown("### 📚 Knowledge Base Topics")
        st.markdown(
            "- Machine Learning fundamentals\n"
            "- Deep Learning & Neural Networks\n"
            "- NLP & Transformers (BERT, GPT)\n"
            "- RAG & Vector Databases\n"
            "- Scikit-learn Pipelines & GridSearch\n"
            "- Model Evaluation Metrics\n"
            "- Transfer Learning & Fine-tuning\n"
            "- Customer Churn Prediction\n"
            "- Embeddings & FAISS"
        )

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.markdown("### 💬 Try these questions")
        example_questions = [
            "What is the difference between supervised and unsupervised learning?",
            "How does BERT work?",
            "What is RAG and why is it useful?",
            "What metrics should I use for imbalanced datasets?",
            "How does FAISS perform similarity search?",
            "What is transfer learning?",
        ]
        for q in example_questions:
            if st.button(q, key=q):
                st.session_state.pending_question = q
                st.rerun()

    # ── Session state ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"human": ..., "ai": ...}
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # ── Build vectorstore (cached) ──
    vectorstore = build_vectorstore()

    if not hf_token:
        st.info("👆 Enter your free HuggingFace token in the sidebar to start chatting.")
        st.stop()

    # ── Display chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Retrieved source chunks"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:** {src[:400]}...")

    # ── Handle input (typed or sidebar button) ──
    prompt = st.chat_input("Ask anything about AI/ML...")

    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating answer..."):
                try:
                    answer, sources = answer_question(
                        prompt, vectorstore, hf_token, st.session_state.chat_history
                    )

                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Retrieved source chunks"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"**Chunk {i}:** {src[:400]}...")

                    # Save to session memory
                    st.session_state.chat_history.append(
                        {"human": prompt, "ai": answer}
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    err = str(e)
                    if "401" in err or "Authorization" in err or "credentials" in err.lower():
                        st.error("❌ Invalid HuggingFace token. Check your token in the sidebar.")
                    elif "429" in err or "rate" in err.lower():
                        st.error("⏳ HuggingFace rate limit reached. Wait ~1 minute and try again.")
                    elif "503" in err or "unavailable" in err.lower():
                        st.error("🔄 HuggingFace model is loading (cold start). Wait 20 seconds and retry.")
                    else:
                        st.error(f"❌ Error: {err}")


if __name__ == "__main__":
    main()
