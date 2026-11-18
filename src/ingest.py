"""
Etapa 1 - Ingestão dos documentos (MLflow Tracking).

Lê os PDFs de data/docs, divide em chunks, gera embeddings e salva o índice
vetorial em data/index. Tudo é registrado como um *run* no MLflow:
  - parâmetros: tamanho do chunk, overlap, modelo de embedding
  - métricas:   nº de documentos, páginas, chunks, tempo de embedding
  - artefatos:  o próprio índice (versionado junto com o run)
"""
import json
import re
import time

import mlflow
import numpy as np
from pypdf import PdfReader

from src.config import (CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, EMBED_MODEL,
                        INDEX_DIR, setup_mlflow)
from src.llm import embed


def load_pages() -> list[dict]:
    """Retorna uma lista de páginas: {source, page, text}. Aceita .pdf, .txt e .md."""
    pages = []
    for path in sorted(DOCS_DIR.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for i, page in enumerate(PdfReader(path).pages, start=1):
                pages.append({"source": path.name, "page": i, "text": page.extract_text() or ""})
        elif suffix in {".txt", ".md"}:
            pages.append({"source": path.name, "page": 1, "text": path.read_text(encoding="utf-8")})
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Divide o texto em pedaços de ~`size` caracteres com sobreposição."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if len(c) > 50]  # descarta pedaços muito pequenos


def main() -> None:
    setup_mlflow()
    pages = load_pages()
    if not pages:
        raise SystemExit(f"Nenhum documento encontrado em {DOCS_DIR}. Coloque seus PDFs lá.")

    chunks = [
        {"source": p["source"], "page": p["page"], "text": c}
        for p in pages
        for c in chunk_text(p["text"], CHUNK_SIZE, CHUNK_OVERLAP)
    ]

    with mlflow.start_run(run_name="ingestao"):
        mlflow.set_tag("etapa", "ingestao")
        mlflow.log_params({
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embed_model": EMBED_MODEL,
        })

        t0 = time.time()
        texts = [c["text"] for c in chunks]
        vectors = np.vstack([embed(texts[i:i + 32]) for i in range(0, len(texts), 32)])
        elapsed = time.time() - t0

        mlflow.log_metrics({
            "num_documentos": len({p["source"] for p in pages}),
            "num_paginas": len(pages),
            "num_chunks": len(chunks),
            "tempo_embedding_s": round(elapsed, 2),
        })

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_DIR / "embeddings.npy", vectors)
        (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
        mlflow.log_artifacts(str(INDEX_DIR), artifact_path="index")

    print(f"OK: {len(pages)} páginas -> {len(chunks)} chunks indexados em {elapsed:.1f}s")


if __name__ == "__main__":
    main()
