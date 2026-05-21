import random
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from experiment_config import DEFAULT_EMBEDDING_MODEL, get_embedding_model_config


def tokenize_for_bm25(text: str) -> list[str]:
    return [token for token in str(text).lower().split() if token]


class ICLRetriever:
    """
    Implements the three shot-selection strategies from the article:

    1. random
    2. similarity-based
    3. diversity-based

    Similarity uses cosine similarity over multilingual sentence embeddings.
    Diversity uses top-N similar candidates, then KMeans to select varied examples.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str = "embeddings",
        cache_name: str = "train_embeddings.npy",
    ):
        self.train_df = train_df.reset_index(drop=True)
        self.embedding_model_name = embedding_model_name
        self.embedding_config = get_embedding_model_config(embedding_model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(embedding_model_name)

        self.cache_path = self.cache_dir / cache_name
        self.embeddings = self._load_or_create_embeddings()

    def _format_for_e5_passage(self, text: str) -> str:
        return f"{self.embedding_config['passage_prefix']}{text}"

    def _format_for_e5_query(self, text: str) -> str:
        return f"{self.embedding_config['query_prefix']}{text}"

    def _load_or_create_embeddings(self) -> np.ndarray:
        if self.cache_path.exists():
            embeddings = np.load(self.cache_path)
            if len(embeddings) == len(self.train_df):
                return embeddings

            print(
                "Embedding cache row count mismatch; "
                f"rebuilding {self.cache_path}"
            )

        texts = [
            self._format_for_e5_passage(str(x))
            for x in self.train_df["source_text"].tolist()
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings)
        np.save(self.cache_path, embeddings)

        return embeddings

    def _encode_query(self, query: str) -> np.ndarray:
        q = self._format_for_e5_query(query)

        emb = self.model.encode(
            [q],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return np.asarray(emb)

    def retrieve_random(
        self,
        k: int = 8,
        seed: int = 42,
        test_id: int = 0,
    ) -> pd.DataFrame:
        rng = random.Random(seed + int(test_id))
        indices = rng.sample(range(len(self.train_df)), k=min(k, len(self.train_df)))
        return self.train_df.iloc[indices].copy()

    def retrieve_similarity(
        self,
        query: str,
        k: int = 8,
    ) -> pd.DataFrame:
        q_emb = self._encode_query(query)

        # embeddings are normalized, so dot product = cosine similarity
        scores = np.dot(self.embeddings, q_emb)

        top_indices = np.argsort(scores)[::-1][:k]

        result = self.train_df.iloc[top_indices].copy()
        result["retrieval_score"] = scores[top_indices]

        return result

    def retrieve_diversity(
        self,
        query: str,
        k: int = 8,
        candidate_n: int = 40,
    ) -> pd.DataFrame:
        q_emb = self._encode_query(query)
        scores = np.dot(self.embeddings, q_emb)

        candidate_n = min(candidate_n, len(self.train_df))
        k = min(k, candidate_n)

        top_indices = np.argsort(scores)[::-1][:candidate_n]
        candidate_embeddings = self.embeddings[top_indices]

        if len(top_indices) <= k:
            result = self.train_df.iloc[top_indices].copy()
            result["retrieval_score"] = scores[top_indices]
            return result

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto",
        )

        labels = kmeans.fit_predict(candidate_embeddings)

        selected_indices = []

        for cluster_id in range(k):
            cluster_positions = np.where(labels == cluster_id)[0]

            if len(cluster_positions) == 0:
                continue

            # выбираем самый похожий пример внутри каждого кластера
            best_position = max(
                cluster_positions,
                key=lambda pos: scores[top_indices[pos]],
            )

            selected_indices.append(top_indices[best_position])

        # если вдруг выбрали меньше k, добираем по similarity
        if len(selected_indices) < k:
            for idx in top_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) == k:
                    break

        result = self.train_df.iloc[selected_indices[:k]].copy()
        result["retrieval_score"] = scores[selected_indices[:k]]

        return result

    def retrieve(
        self,
        query: str,
        strategy: str,
        k: int = 8,
        seed: int = 42,
        test_id: int = 0,
        candidate_n: int = 40,
    ) -> pd.DataFrame:
        strategy = strategy.lower().strip()

        if strategy == "random":
            return self.retrieve_random(k=k, seed=seed, test_id=test_id)

        if strategy == "similarity":
            return self.retrieve_similarity(query=query, k=k)

        if strategy == "diversity":
            return self.retrieve_diversity(
                query=query,
                k=k,
                candidate_n=candidate_n,
            )

        raise ValueError(f"Unknown retrieval strategy: {strategy}")


class BM25ICLRetriever:
    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df.reset_index(drop=True)
        self.documents = [
            tokenize_for_bm25(text)
            for text in self.train_df["source_text"].astype(str).tolist()
        ]
        self.index = BM25Okapi(self.documents)
        self.token_sets = [
            set(tokenize_for_bm25(text))
            for text in self.train_df["source_text"].astype(str).tolist()
        ]

    def retrieve_random(
        self,
        k: int = 8,
        seed: int = 42,
        test_id: int = 0,
    ) -> pd.DataFrame:
        rng = random.Random(seed + int(test_id))
        indices = rng.sample(range(len(self.train_df)), k=min(k, len(self.train_df)))
        return self.train_df.iloc[indices].copy()

    def retrieve_similarity(
        self,
        query: str,
        k: int = 8,
    ) -> pd.DataFrame:
        scores = np.asarray(self.index.get_scores(tokenize_for_bm25(query)))
        top_indices = np.argsort(scores)[::-1][:k]

        result = self.train_df.iloc[top_indices].copy()
        result["retrieval_score"] = scores[top_indices]
        return result

    def retrieve_diversity(
        self,
        query: str,
        k: int = 8,
        candidate_n: int = 40,
    ) -> pd.DataFrame:
        scores = np.asarray(self.index.get_scores(tokenize_for_bm25(query)))
        candidate_n = min(candidate_n, len(self.train_df))
        k = min(k, candidate_n)
        top_indices = np.argsort(scores)[::-1][:candidate_n].tolist()

        if len(top_indices) <= k:
            result = self.train_df.iloc[top_indices].copy()
            result["retrieval_score"] = scores[top_indices]
            return result

        selected_indices = [top_indices[0]]
        max_score = float(scores[top_indices[0]]) or 1.0

        while len(selected_indices) < k:
            best_idx = None
            best_score = None

            for idx in top_indices:
                if idx in selected_indices:
                    continue

                overlap_penalty = max(
                    self._jaccard(idx, selected_idx)
                    for selected_idx in selected_indices
                )
                normalized_score = float(scores[idx]) / max_score
                diversity_score = normalized_score - 0.25 * overlap_penalty

                if best_score is None or diversity_score > best_score:
                    best_score = diversity_score
                    best_idx = idx

            if best_idx is None:
                break

            selected_indices.append(best_idx)

        result = self.train_df.iloc[selected_indices[:k]].copy()
        result["retrieval_score"] = scores[selected_indices[:k]]
        return result

    def _jaccard(self, left_idx: int, right_idx: int) -> float:
        left = self.token_sets[left_idx]
        right = self.token_sets[right_idx]

        if not left and not right:
            return 0.0

        return len(left & right) / max(1, len(left | right))

    def retrieve(
        self,
        query: str,
        strategy: str,
        k: int = 8,
        seed: int = 42,
        test_id: int = 0,
        candidate_n: int = 40,
    ) -> pd.DataFrame:
        strategy = strategy.lower().strip()

        if strategy == "random":
            return self.retrieve_random(k=k, seed=seed, test_id=test_id)

        if strategy == "similarity":
            return self.retrieve_similarity(query=query, k=k)

        if strategy == "diversity":
            return self.retrieve_diversity(
                query=query,
                k=k,
                candidate_n=candidate_n,
            )

        raise ValueError(f"Unknown retrieval strategy: {strategy}")
