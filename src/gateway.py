"""
Etapa 0 - Configuração do MLflow AI Gateway.

Cadastra o Ollama local dentro do AI Gateway embutido no servidor do MLflow 3:
  1. Secret ("ollama-local") com a URL base do container do Ollama
  2. Model Definitions para o LLM de chat e o modelo de embeddings
  3. Dois Endpoints lógicos no Gateway:
       - chat-mlops  -> aponta para o LLM_MODEL (ex.: qwen2.5:1.5b)
       - embed-mlops -> aponta para o EMBED_MODEL (ex.: nomic-embed-text)

Assim, a aplicação (src/llm.py) não chama o Ollama diretamente: ela chama
sempre o AI Gateway do MLflow (http://mlflow:5000/gateway/openai/v1) usando os
nomes lógicos "chat-mlops" e "embed-mlops".
"""
import json
import urllib.request

from src.config import (CHAT_ENDPOINT, EMBED_ENDPOINT, EMBED_MODEL, LLM_MODEL,
                        MLFLOW_TRACKING_URI, OLLAMA_BASE_URL)

API = f"{MLFLOW_TRACKING_URI}/api/3.0/mlflow/gateway"


def _call(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{API}/{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def ensure_secret(name: str = "ollama-local") -> str:
    for s in _call("GET", "secrets/list").get("secrets", []):
        if s["secret_name"] == name:
            return s["secret_id"]
    res = _call("POST", "secrets/create", {
        "secret_name": name,
        "provider": "ollama",
        "secret_value": {"api_key": "ollama"},
        "auth_config": {"api_base": OLLAMA_BASE_URL},
    })
    return res["secret"]["secret_id"]


def ensure_model_def(name: str, model_name: str, secret_id: str) -> str:
    for m in _call("GET", "model-definitions/list").get("model_definitions", []):
        if m["name"] == name:
            if m["model_name"] != model_name:
                _call("POST", "model-definitions/update", {
                    "model_definition_id": m["model_definition_id"],
                    "model_name": model_name,
                })
            return m["model_definition_id"]
    res = _call("POST", "model-definitions/create", {
        "name": name,
        "secret_id": secret_id,
        "provider": "ollama",
        "model_name": model_name,
    })
    return res["model_definition"]["model_definition_id"]


def ensure_endpoint(endpoint_name: str, model_def_id: str) -> None:
    for ep in _call("GET", "endpoints/list").get("endpoints", []):
        if ep["name"] == endpoint_name:
            print(f"Endpoint '{endpoint_name}' já está ativo no MLflow AI Gateway.")
            return
    _call("POST", "endpoints/create", {
        "name": endpoint_name,
        "usage_tracking": True,
        "model_configs": [{
            "model_definition_id": model_def_id,
            "linkage_type": "PRIMARY",
            "weight": 1.0,
        }],
    })
    print(f"Endpoint '{endpoint_name}' criado no MLflow AI Gateway.")


def main() -> None:
    secret_id = ensure_secret()
    chat_def = ensure_model_def("ollama-chat-model", LLM_MODEL, secret_id)
    embed_def = ensure_model_def("ollama-embed-model", EMBED_MODEL, secret_id)

    ensure_endpoint(CHAT_ENDPOINT, chat_def)
    ensure_endpoint(EMBED_ENDPOINT, embed_def)
    print(f"OK: AI Gateway roteando '{CHAT_ENDPOINT}' -> {LLM_MODEL} e '{EMBED_ENDPOINT}' -> {EMBED_MODEL}")


if __name__ == "__main__":
    main()
