"""
Etapa 3 - Avaliação (MLflow GenAI Evaluation).

Roda o RAG sobre um pequeno conjunto de perguntas de teste (data/eval_set.json)
e calcula métricas com *scorers*:
  - cobertura_palavras_chave: % das palavras-chave esperadas que aparecem na resposta
  - comportamento_esperado: respondeu quando devia e recusou ("Não encontrei...")
                            quando a pergunta está fora do material (alucinação)
  - recuperou_fonte:        a busca (retrieval) trouxe a página certa do PDF?
  - juiz_llm:               "LLM-as-a-judge" usando o próprio LLM local para julgar a resposta
Os resultados (por pergunta e agregados) aparecem na aba "Evaluations" do MLflow.
"""
import json
import unicodedata

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from src.config import EVAL_FILE, LLM_MODEL, PROMPT_ALIAS, PROMPT_NAME, TOP_K, setup_mlflow
from src.llm import chat
from src.rag import RAG


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


@scorer
def cobertura_palavras_chave(outputs: dict, expectations: dict) -> float:
    answer = _norm(outputs["answer"])
    keywords = expectations["palavras_chave"]
    return sum(_norm(k) in answer for k in keywords) / len(keywords)


@scorer
def comportamento_esperado(outputs: dict, expectations: dict) -> bool:
    respondeu = "não encontrei" not in _norm(outputs["answer"])
    return respondeu == expectations.get("deve_responder", True)


@scorer
def recuperou_fonte(outputs: dict, expectations: dict) -> Feedback | None:
    if "fonte" not in expectations:  # pergunta fora do material: não se aplica
        return None
    esperadas = {f"{_norm(expectations['fonte'])} (p.{p})" for p in expectations["paginas"]}
    obtidas = {_norm(s) for s in outputs["sources"]}
    return Feedback(value=bool(esperadas & obtidas), rationale=f"Recuperadas: {sorted(obtidas)}")


JUDGE_PROMPT = """Você é um avaliador. Compare a RESPOSTA GERADA com a RESPOSTA ESPERADA.
A resposta gerada está correta (mesmo que com outras palavras)?
Responda na primeira linha apenas SIM ou NAO e, na segunda linha, uma justificativa curta.

PERGUNTA: {question}
RESPOSTA ESPERADA: {expected}
RESPOSTA GERADA: {answer}"""


@scorer
def juiz_llm(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    verdict = chat([{
        "role": "user",
        "content": JUDGE_PROMPT.format(
            question=inputs["question"],
            expected=expectations["resposta_esperada"],
            answer=outputs["answer"],
        ),
    }], temperature=0.0)
    first_line, _, rationale = verdict.partition("\n")
    return Feedback(value=first_line.strip().upper().startswith("SIM"), rationale=rationale.strip() or verdict)


def main() -> None:
    setup_mlflow()
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    rag = RAG()

    def predict_fn(question: str) -> dict:
        return rag.answer(question)

    with mlflow.start_run(run_name="avaliacao"):
        mlflow.set_tag("etapa", "avaliacao")
        prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}")
        mlflow.log_params({"llm_model": LLM_MODEL, "top_k": TOP_K, "prompt_version": prompt.version})

        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=[cobertura_palavras_chave, comportamento_esperado, recuperou_fonte, juiz_llm],
        )

    print("\nMétricas agregadas:")
    for name, value in results.metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
