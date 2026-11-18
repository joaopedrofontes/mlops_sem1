"""
Definição do modelo para o MLflow ("Models from Code").

Empacota o pipeline RAG inteiro como um modelo pyfunc: o índice vetorial vai
junto como artefato. O prompt continua vindo do Prompt Registry e o LLM do Ollama.
"""
from pathlib import Path

import mlflow
from mlflow.pyfunc import PythonModel


class RAGModel(PythonModel):
    def load_context(self, context):
        from src.rag import RAG

        self.rag = RAG(index_dir=Path(context.artifacts["index"]))

    def predict(self, context, model_input: list[str], params=None) -> list[str]:
        return [self.rag.answer(q)["answer"] for q in model_input]


mlflow.models.set_model(RAGModel())
