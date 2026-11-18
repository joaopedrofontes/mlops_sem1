"""
Interface de chat (Streamlit) + coleta de Feedback humano.

Cada pergunta gera um trace no MLflow. Os botões 👍/👎 gravam o feedback do
usuário *dentro do trace* com `mlflow.log_feedback`, fechando o ciclo de
monitoramento: dá para filtrar no MLflow UI as respostas mal avaliadas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import streamlit as st
from mlflow.entities import AssessmentSource, AssessmentSourceType

from src.config import LLM_MODEL, setup_mlflow
from src.rag import RAG

st.set_page_config(page_title="Assistente de MLOps", page_icon="🤖")
setup_mlflow()


@st.cache_resource
def get_rag() -> RAG:
    return RAG()


def send_feedback(msg: dict, key: str) -> None:
    value = st.session_state[key]  # 1 = 👍, 0 = 👎
    mlflow.log_feedback(
        trace_id=msg["trace_id"],
        name="feedback_usuario",
        value=bool(value),
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="usuario-streamlit"),
    )


st.title("🤖 Assistente da disciplina de MLOps")
st.caption(f"RAG local com Ollama ({LLM_MODEL}) + MLflow (tracing, prompt registry e feedback)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander("Fontes"):
                st.markdown("\n".join(f"- {s}" for s in msg["sources"]))
            key = f"fb_{i}"
            st.feedback("thumbs", key=key, on_change=send_feedback, args=(msg, key))

if question := st.chat_input("Pergunte algo sobre o material da disciplina..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Pensando..."):
        result = get_rag().answer(question)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "trace_id": mlflow.get_last_active_trace_id(),
    })
    st.rerun()
