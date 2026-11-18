"""
Pipeline RAG com MLflow Tracing.

- `mlflow.openai.autolog()` captura automaticamente cada chamada ao LLM
  (prompt, resposta, tokens, latência).
- `@mlflow.trace` cria spans para as nossas funções (retrieve / answer),
  então no MLflow UI cada pergunta vira um trace com a árvore completa.
"""
import json
from pathlib import Path

import mlflow
import numpy as np
from mlflow.entities import SpanType

from src.config import INDEX_DIR, PROMPT_ALIAS, PROMPT_NAME, TOP_K
from src.llm import chat, embed

mlflow.openai.autolog()


class RAG:
    def __init__(self, index_dir: Path = INDEX_DIR):
        index_dir = Path(index_dir)
        if not (index_dir / "embeddings.npy").exists():
            raise FileNotFoundError("Índice não encontrado. Rode a ingestão primeiro (make ingest).")
        self.embeddings = np.load(index_dir / "embeddings.npy")
        self.chunks = json.loads((index_dir / "chunks.json").read_text())

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        """Busca os k chunks mais similares (similaridade de cosseno)."""
        q = embed([question])[0]
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        # Formato {page_content, metadata} é exibido de forma especial no MLflow UI
        return [
            {
                "page_content": self.chunks[i]["text"],
                "metadata": {
                    "source": self.chunks[i]["source"],
                    "page": self.chunks[i]["page"],
                    "score": round(float(scores[i]), 4),
                },
            }
            for i in top
        ]

    @mlflow.trace(span_type=SpanType.CHAIN)
    def answer(self, question: str) -> dict:
        docs = self.retrieve(question)
        context = "\n\n".join(
            f"[{d['metadata']['source']} - p.{d['metadata']['page']}]\n{d['page_content']}"
            for d in docs
        )
        # Prompt vem do Prompt Registry (versão apontada pelo alias)
        prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}")
        answer = chat([{"role": "user", "content": prompt.format(context=context, question=question)}])

        sources = sorted({f"{d['metadata']['source']} (p.{d['metadata']['page']})" for d in docs})
        return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys

    from src.config import setup_mlflow

    setup_mlflow()
    question = " ".join(sys.argv[1:]) or "O que é MLOps?"
    result = RAG().answer(question)
    print(result["answer"], "\n\nFontes:", ", ".join(result["sources"]))
