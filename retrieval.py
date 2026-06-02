import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from experiment_config import DEFAULT_EMBEDDING_MODEL, get_embedding_model_config


def tokenize_for_bm25(text: str) -> list[str]:
    return [token for token in str(text).lower().split() if token]


def normalize_graph_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def tokenize_for_graph(text: str) -> list[str]:
    text = normalize_graph_text(text)
    return [token for token in re.split(r"[^\wʿʾ'-]+", text) if token]


def length_bucket(length_value, fallback_tokens: list[str]) -> str:
    try:
        length = int(float(length_value))
    except Exception:
        length = len(fallback_tokens)

    if length <= 1:
        return "1"
    if length <= 2:
        return "2"
    if length <= 4:
        return "3-4"
    if length <= 8:
        return "5-8"
    return "9+"


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if math.isclose(min_score, max_score):
        return np.zeros_like(scores, dtype=float)

    return (scores - min_score) / (max_score - min_score)


class LocalDenseVectorStore:
    """
    Minimal local vector store backed by cached sentence-transformer embeddings.

    This keeps the first graph experiment lightweight: no external vector DB is
    needed, but the retrieval code still has an explicit vector-store boundary.
    """

    def __init__(
        self,
        texts: list[str],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str = "embeddings",
        cache_name: str = "train_embeddings.npy",
    ):
        self.texts = [str(text) for text in texts]
        self.embedding_model_name = embedding_model_name
        self.embedding_config = get_embedding_model_config(embedding_model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / cache_name

        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self._load_or_create_embeddings()

    def _format_for_passage(self, text: str) -> str:
        return f"{self.embedding_config['passage_prefix']}{text}"

    def _format_for_query(self, text: str) -> str:
        return f"{self.embedding_config['query_prefix']}{text}"

    def _load_or_create_embeddings(self) -> np.ndarray:
        if self.cache_path.exists():
            embeddings = np.load(self.cache_path)
            if len(embeddings) == len(self.texts):
                return embeddings

            print(
                "Vector cache row count mismatch; "
                f"rebuilding {self.cache_path}"
            )

        texts = [self._format_for_passage(text) for text in self.texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings)
        np.save(self.cache_path, embeddings)

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            [self._format_for_query(query)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(emb)

    def score(self, query: str) -> np.ndarray:
        q_emb = self.encode_query(query)
        return np.dot(self.embeddings, q_emb)


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


class GraphICLRetriever:
    """
    Graph-aware in-context example retriever.

    The graph is a lightweight heterogeneous feature graph:
    source examples connect to lexical, morphology-ish, type, length, and sheet
    feature nodes. At query time we add a temporary query node via the same
    feature extraction and rank train examples with one of three strategies:

    1. graph_common: weighted common-neighbor score
    2. graph_ppr: personalized PageRank-style walk from query features
    3. hybrid_graph: dense vector score + graph_ppr + graph_common
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str = "embeddings",
        cache_name: str = "graph_dense_embeddings.npy",
        ppr_steps: int = 8,
        restart_prob: float = 0.35,
    ):
        self.train_df = train_df.reset_index(drop=True)
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.ppr_steps = max(1, int(ppr_steps))
        self.restart_prob = min(max(float(restart_prob), 0.05), 0.95)
        self.vector_store = None

        self.raw_source_features = []
        self.source_features = []
        self.feature_idf = {}
        self.feature_to_sources = defaultdict(list)
        self.source_to_features = []
        self.feature_totals = {}
        self.source_totals = []

        self._build_graph()

    def _row_features(self, row: dict) -> dict[str, float]:
        source_text = row.get("source_text", "")
        tokens = tokenize_for_graph(source_text)
        features = defaultdict(float)

        for token in tokens:
            features[f"token:{token}"] += 3.0

            if len(token) >= 3:
                features[f"suffix3:{token[-3:]}"] += 1.2
                features[f"prefix3:{token[:3]}"] += 0.6

            if len(token) >= 4:
                features[f"suffix4:{token[-4:]}"] += 0.9

        for left, right in zip(tokens, tokens[1:], strict=False):
            features[f"bigram:{left} {right}"] += 2.2

        row_type = str(row.get("type", "")).strip().lower()
        if row_type:
            features[f"type:{row_type}"] += 0.8

        bucket = length_bucket(row.get("length", ""), fallback_tokens=tokens)
        features[f"length:{bucket}"] += 0.6

        sheet = str(row.get("sheet", "")).strip()
        if sheet:
            features[f"sheet:{sheet}"] += 0.4

        return dict(features)

    def _build_graph(self):
        doc_counts = defaultdict(int)

        for _, row in self.train_df.iterrows():
            raw_features = self._row_features(row.to_dict())
            self.raw_source_features.append(raw_features)

            for feature in raw_features:
                doc_counts[feature] += 1

        n_docs = max(1, len(self.train_df))
        self.feature_idf = {
            feature: math.log((n_docs + 1) / (count + 1)) + 1.0
            for feature, count in doc_counts.items()
        }

        for source_idx, raw_features in enumerate(self.raw_source_features):
            weighted = {
                feature: weight * self.feature_idf.get(feature, 1.0)
                for feature, weight in raw_features.items()
            }
            self.source_features.append(weighted)
            source_edges = list(weighted.items())
            self.source_to_features.append(source_edges)
            self.source_totals.append(sum(weighted.values()) or 1.0)

            for feature, weight in source_edges:
                self.feature_to_sources[feature].append((source_idx, weight))

        self.feature_totals = {
            feature: sum(weight for _, weight in edges) or 1.0
            for feature, edges in self.feature_to_sources.items()
        }

    def _query_features(
        self,
        query: str,
        query_metadata: dict | None = None,
    ) -> dict[str, float]:
        row = {"source_text": query}
        if query_metadata:
            row.update(query_metadata)
            row["source_text"] = query

        raw_features = self._row_features(row)
        return {
            feature: weight * self.feature_idf.get(feature, 1.0)
            for feature, weight in raw_features.items()
        }

    def _rank_result(self, scores: np.ndarray, k: int) -> pd.DataFrame:
        k = min(int(k), len(self.train_df))
        top_indices = np.argsort(scores)[::-1][:k]
        result = self.train_df.iloc[top_indices].copy()
        result["retrieval_score"] = scores[top_indices]
        return result

    def retrieve_random(
        self,
        k: int = 8,
        seed: int = 42,
        test_id: int = 0,
    ) -> pd.DataFrame:
        rng = random.Random(seed + int(test_id))
        indices = rng.sample(range(len(self.train_df)), k=min(k, len(self.train_df)))
        return self.train_df.iloc[indices].copy()

    def graph_common_scores(
        self,
        query: str,
        query_metadata: dict | None = None,
    ) -> np.ndarray:
        query_features = self._query_features(query, query_metadata=query_metadata)
        scores = np.zeros(len(self.train_df), dtype=float)
        query_norm = math.sqrt(
            sum(weight * weight for weight in query_features.values())
        )

        for source_idx, source_features in enumerate(self.source_features):
            common = set(query_features) & set(source_features)
            if not common:
                continue

            dot = sum(query_features[f] * source_features[f] for f in common)
            source_norm = math.sqrt(
                sum(weight * weight for weight in source_features.values())
            )
            scores[source_idx] = dot / max(query_norm * source_norm, 1e-12)

        return scores

    def graph_ppr_scores(
        self,
        query: str,
        query_metadata: dict | None = None,
    ) -> np.ndarray:
        query_features = self._query_features(query, query_metadata=query_metadata)
        seen_query_features = {
            feature: weight
            for feature, weight in query_features.items()
            if feature in self.feature_to_sources
        }

        total = sum(seen_query_features.values()) or 1.0
        restart_dist = {
            feature: weight / total
            for feature, weight in seen_query_features.items()
        }

        if not restart_dist:
            return self.graph_common_scores(query, query_metadata=query_metadata)

        feature_dist = dict(restart_dist)
        source_dist = np.zeros(len(self.train_df), dtype=float)

        for _ in range(self.ppr_steps):
            next_source_dist = np.zeros(len(self.train_df), dtype=float)

            for feature, prob in feature_dist.items():
                for source_idx, weight in self.feature_to_sources.get(feature, []):
                    next_source_dist[source_idx] += (
                        prob * weight / self.feature_totals[feature]
                    )

            next_feature_dist = defaultdict(float)

            for source_idx, prob in enumerate(next_source_dist):
                if prob == 0:
                    continue

                for feature, weight in self.source_to_features[source_idx]:
                    next_feature_dist[feature] += (
                        prob * weight / self.source_totals[source_idx]
                    )

            source_dist = next_source_dist
            feature_dist = defaultdict(float)

            for feature, prob in next_feature_dist.items():
                feature_dist[feature] += (1.0 - self.restart_prob) * prob

            for feature, prob in restart_dist.items():
                feature_dist[feature] += self.restart_prob * prob

        return source_dist

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.vector_store is None:
            self.vector_store = LocalDenseVectorStore(
                texts=self.train_df["source_text"].astype(str).tolist(),
                embedding_model_name=self.embedding_model_name,
                cache_dir=self.cache_dir,
                cache_name=self.cache_name,
            )

        return self.vector_store.score(query)

    def hybrid_graph_scores(
        self,
        query: str,
        query_metadata: dict | None = None,
    ) -> np.ndarray:
        dense_scores = normalize_scores(self._dense_scores(query))
        ppr_scores = normalize_scores(
            self.graph_ppr_scores(query, query_metadata=query_metadata)
        )
        common_scores = normalize_scores(
            self.graph_common_scores(query, query_metadata=query_metadata)
        )

        return 0.55 * dense_scores + 0.30 * ppr_scores + 0.15 * common_scores

    def retrieve_graph_common(
        self,
        query: str,
        k: int = 8,
        query_metadata: dict | None = None,
    ) -> pd.DataFrame:
        scores = self.graph_common_scores(query, query_metadata=query_metadata)
        return self._rank_result(scores, k=k)

    def retrieve_graph_ppr(
        self,
        query: str,
        k: int = 8,
        query_metadata: dict | None = None,
    ) -> pd.DataFrame:
        scores = self.graph_ppr_scores(query, query_metadata=query_metadata)
        return self._rank_result(scores, k=k)

    def retrieve_hybrid_graph(
        self,
        query: str,
        k: int = 8,
        query_metadata: dict | None = None,
    ) -> pd.DataFrame:
        scores = self.hybrid_graph_scores(query, query_metadata=query_metadata)
        return self._rank_result(scores, k=k)

    def retrieve(
        self,
        query: str,
        strategy: str,
        k: int = 8,
        seed: int = 42,
        test_id: int = 0,
        candidate_n: int = 40,
        query_metadata: dict | None = None,
    ) -> pd.DataFrame:
        strategy = strategy.lower().strip()

        if strategy == "random":
            return self.retrieve_random(k=k, seed=seed, test_id=test_id)

        if strategy == "graph_common":
            return self.retrieve_graph_common(
                query=query,
                k=k,
                query_metadata=query_metadata,
            )

        if strategy == "graph_ppr":
            return self.retrieve_graph_ppr(
                query=query,
                k=k,
                query_metadata=query_metadata,
            )

        if strategy == "hybrid_graph":
            return self.retrieve_hybrid_graph(
                query=query,
                k=k,
                query_metadata=query_metadata,
            )

        raise ValueError(f"Unknown graph retrieval strategy: {strategy}")


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
