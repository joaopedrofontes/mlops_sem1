"""Configurações centrais do projeto (lidas de variáveis de ambiente / .env)."""
import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "index"
EVAL_FILE = ROOT / "data" / "eval_set.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MLFLOW_GATEWAY_URL = os.getenv("MLFLOW_GATEWAY_URL", f"{MLFLOW_TRACKING_URI}/gateway/openai/v1")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT", "chat-mlops")
EMBED_ENDPOINT = os.getenv("EMBED_ENDPOINT", "embed-mlops")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Nomes dos recursos no MLflow
EXPERIMENT_NAME = "rag-mlops"
PROMPT_NAME = "rag-mlops-prompt"
PROMPT_ALIAS = "production"
MODEL_NAME = "rag-mlops"


def setup_mlflow() -> None:
    """Aponta para o servidor MLflow e seleciona o experimento do projeto."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
