"""
Cliente de LLM e Embeddings via MLflow AI Gateway.

A aplicação NÃO se conecta diretamente ao Ollama e nem precisa saber qual modelo
físico está rodando: ela envia requisições OpenAI-compatíveis para o AI Gateway
do MLflow (`http://mlflow:5000/gateway/openai/v1`) chamando os endpoints lógicos
`chat-mlops` e `embed-mlops`, e o Gateway roteia para o provedor configurado.
"""
import numpy as np
from openai import OpenAI

from src.config import CHAT_ENDPOINT, EMBED_ENDPOINT, MLFLOW_GATEWAY_URL

client = OpenAI(base_url=MLFLOW_GATEWAY_URL, api_key="mlflow", timeout=180, max_retries=1)


def embed(texts: list[str]) -> np.ndarray:
    """Gera embeddings normalizados (norma 1) via endpoint do MLflow AI Gateway."""
    resp = client.embeddings.create(model=EMBED_ENDPOINT, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    """Envia mensagens ao endpoint de chat do MLflow AI Gateway."""
    resp = client.chat.completions.create(
        model=CHAT_ENDPOINT, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()
