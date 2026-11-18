"""
Etapa 2 - Prompt Registry.

Registra o template do prompt no MLflow (com versionamento) e aponta o alias
"production" para a versão atual. A aplicação sempre carrega
`prompts:/rag-mlops-prompt@production`, então para trocar o prompt em produção
basta editar o TEMPLATE abaixo e rodar este script de novo: uma nova versão é
criada e o alias passa a apontar para ela, sem mudar nada no código do app.
"""
import mlflow

from src.config import PROMPT_ALIAS, PROMPT_NAME, setup_mlflow

TEMPLATE = """Você é um assistente da disciplina de MLOps.
Responda à pergunta usando APENAS as informações do contexto abaixo.
Se a resposta não estiver no contexto, responda exatamente:
"Não encontrei essa informação no material da disciplina."
Responda em português, de forma clara e objetiva.

Contexto:
{{context}}

Pergunta: {{question}}"""


def main() -> None:
    setup_mlflow()

    # Evita criar versões duplicadas se o template não mudou
    try:
        current = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}")
        if current.template == TEMPLATE:
            print(f"Prompt inalterado (versão {current.version} continua em '{PROMPT_ALIAS}').")
            return
    except Exception:
        pass  # primeira execução: o prompt ainda não existe

    prompt = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=TEMPLATE,
        commit_message="Prompt do assistente RAG da disciplina",
        tags={"tarefa": "rag", "idioma": "pt-br"},
    )
    mlflow.genai.set_prompt_alias(PROMPT_NAME, alias=PROMPT_ALIAS, version=prompt.version)
    print(f"Prompt '{PROMPT_NAME}' versão {prompt.version} registrado com alias '{PROMPT_ALIAS}'.")


if __name__ == "__main__":
    main()
