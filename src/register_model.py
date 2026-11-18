"""
Etapa 4 - Model Registry.

Registra o pipeline RAG como modelo versionado ("rag-mlops") e aponta o alias
"champion" para a nova versão. O serviço `serve` (mlflow models serve) sempre
carrega models:/rag-mlops@champion.
"""
import mlflow
from mlflow import MlflowClient

from src.config import INDEX_DIR, MODEL_NAME, ROOT, setup_mlflow


def main() -> None:
    setup_mlflow()
    with mlflow.start_run(run_name="registro-modelo"):
        mlflow.set_tag("etapa", "registro")
        info = mlflow.pyfunc.log_model(
            name="rag_model",
            python_model=str(ROOT / "src" / "model.py"),
            artifacts={"index": str(INDEX_DIR)},
            code_paths=[str(ROOT / "src")],
            pip_requirements=str(ROOT / "requirements.txt"),
            input_example=["O que é MLOps?"],
            registered_model_name=MODEL_NAME,
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"Modelo '{MODEL_NAME}' versão {version} registrado com alias 'champion'.")


if __name__ == "__main__":
    main()
