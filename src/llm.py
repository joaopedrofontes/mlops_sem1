"""Cliente do LLM local (Ollama expõe uma API compatível com a da OpenAI)."""
import numpy as np
from openai import OpenAI

from src.config import EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL

# A chave é ignorada pelo Ollama; timeout evita esperar 10 min se algo travar
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180, max_retries=1)


def embed(texts: list[str]) -> np.ndarray:
    """Gera embeddings normalizados (norma 1) para uma lista de textos."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    """Envia mensagens ao LLM e devolve o texto da resposta."""
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()
