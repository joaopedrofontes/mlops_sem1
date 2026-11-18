RUN = docker compose run --rm app python -m

.PHONY: up down ingest prompt ask eval register serve logs

up:        ## Sobe Ollama, MLflow e o chat (baixa os modelos na 1ª vez)
	docker compose up -d ollama mlflow
	docker compose up ollama-pull
	docker compose up -d app

ingest:    ## Etapa 1 - indexa os PDFs de data/docs
	$(RUN) src.ingest

prompt:    ## Etapa 2 - registra/atualiza o prompt no Prompt Registry
	$(RUN) src.prompts

ask:       ## Pergunta rápida pelo terminal: make ask Q="O que é MLOps?"
	$(RUN) src.rag "$(Q)"

eval:      ## Etapa 3 - avaliação com mlflow.genai.evaluate
	$(RUN) src.evaluate

register:  ## Etapa 4 - registra o modelo no Model Registry (alias champion)
	$(RUN) src.register_model

serve:     ## Etapa 5 - sobe a API REST do modelo champion na porta 5001
	docker compose --profile serve up -d serve

down:
	docker compose --profile serve down

reset:     ## Apaga runs/traces/prompts/modelos do MLflow e o índice (mantém os modelos do Ollama)
	docker compose --profile serve down
	docker volume rm -f sem1_mlflow-data
	rm -rf data/index

logs:
	docker compose logs -f app
