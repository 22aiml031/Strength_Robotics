from __future__ import annotations

import requests


class OllamaEmbeddingClient:
    def __init__(self, base_url: str, model: str, expected_dim: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.expected_dim = expected_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )

        if response.status_code in {400, 404}:
            return [self._embed_one_legacy(text) for text in texts]

        if not response.ok:
            raise RuntimeError(f"Ollama embed failed: {response.status_code} {response.text}")
        payload = response.json()
        embeddings = payload.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"Ollama embed response did not include embeddings: {payload}")

        self._validate_dimensions(embeddings)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed_one_legacy(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(
                f"Ollama legacy embeddings failed for model '{self.model}': "
                f"{response.status_code} {response.text}"
            )
        embedding = response.json().get("embedding")
        if embedding is None:
            raise RuntimeError("Ollama legacy embeddings response did not include embedding.")
        self._validate_dimensions([embedding])
        return embedding

    def _validate_dimensions(self, embeddings: list[list[float]]) -> None:
        for embedding in embeddings:
            if len(embedding) != self.expected_dim:
                raise ValueError(
                    f"Embedding model '{self.model}' returned {len(embedding)} dimensions, "
                    f"but EMBEDDING_DIM is {self.expected_dim}. Update .env and sql/schema.sql "
                    "to use the same dimension."
                )
