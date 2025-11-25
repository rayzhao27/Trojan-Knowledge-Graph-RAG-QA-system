import logging

from typing import List, Dict, Tuple
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from ..utils.cache import cached_search, cache
from ..utils.circuit_breaker import search_circuit_breaker, rerank_circuit_breaker


logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, alpha: float = 0.5, use_rerank: bool = False, use_graph: bool = True):
        self.alpha = alpha
        self.use_rerank = use_rerank
        self.use_graph = use_graph
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.graph_retriever = GraphRetriever() if use_graph else None
        self.reranker = self._init_reranker() if use_rerank else None

        logger.info("HybridRetriever initialized with PyTorch")
        logger.info(f"Search balance: {alpha:.1f} vector + {1 - alpha:.1f} BM25")
        if use_graph:
            logger.info("Graph-based retrieval enabled")

    def _init_reranker(self):
        """Initialize cross-encoder reranker"""
        try:
            from sentence_transformers import CrossEncoder
            print("[INFO] Cross-encoder reranking enabled")
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            print("[WARN] Cross-encoder not available, disabling reranking")
            self.use_rerank = False
            return None

    def build_index(self, chunks: List[Dict], index_file: str = None):
        """Build both indices with PyTorch optimization"""
        logger.info("Building hybrid retrieval index...")

        self.vector_retriever.build_index(chunks, index_file=index_file)
        self.bm25_retriever.build_index(chunks)

        if self.graph_retriever and self.vector_retriever.index:
            # Get embeddings for graph building
            texts = [chunk['text'] for chunk in chunks]
            import torch
            with torch.no_grad():
                embeddings = self.vector_retriever.model.encode(
                    texts,
                    convert_to_numpy=True,
                    device=self.vector_retriever.device,
                    show_progress_bar=True,
                    num_workers=0
                )
            # Normalize embeddings
            import numpy as np
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / norms
            
            self.graph_retriever.build_graph(chunks, embeddings_normalized)

        device_info = self.vector_retriever.get_device_info()
        logger.info("Hybrid retriever ready!")
        logger.info("Device: {device_info['device']}")
        logger.info("PyTorch: {device_info['pytorch_version']}")
        if device_info['cuda_available']:
            logger.info(f"CUDA GPUs: {device_info['cuda_device_count']}")

    @cached_search
    @search_circuit_breaker
    def search(self, query: str, k: int = 10, use_rerank: bool = None, use_graph: bool = None) -> List[Tuple[Dict, float]]:
        """Hybrid search with optional cross-encoder reranking and graph expansion"""
        should_rerank = use_rerank if use_rerank is not None else self.use_rerank
        should_use_graph = use_graph if use_graph is not None else self.use_graph
        search_k = k * 3 if should_rerank else k * 2

        # Get hybrid results
        hybrid_results = self._get_hybrid_results(query, search_k)

        # Apply reranking if needed
        if should_rerank and self.reranker:
            hybrid_results = self._rerank_results(query, hybrid_results, k * 2)
        
        # Apply graph expansion if enabled
        if should_use_graph and self.graph_retriever:
            hybrid_results = self.graph_retriever.expand_results(
                hybrid_results, 
                max_hops=1, 
                max_total=k * 2
            )

        return hybrid_results[:k]

    def _get_hybrid_results(self, query: str, search_k: int) -> List[Tuple[Dict, float]]:
        """Get combined results from vector and BM25 search"""
        vector_results = self.vector_retriever.search(query, search_k)
        bm25_results = self.bm25_retriever.search(query, search_k)

        # Normalize scores
        vector_scores = self._extract_and_normalize_scores(vector_results)
        bm25_scores = self._extract_and_normalize_scores(bm25_results)

        # Combine scores
        combined_scores = self._combine_scores(
            vector_results, bm25_results,
            vector_scores, bm25_scores
        )

        # Sort and return
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return [(item['chunk'], item['score']) for item in sorted_results]

    def _extract_and_normalize_scores(self, results: List[Tuple[Dict, float]]) -> List[float]:
        """Extract scores from results and normalize them"""
        scores = [score for _, score in results]
        return self._normalize_scores_pytorch(scores)

    def _combine_scores(
            self,
            vector_results: List[Tuple[Dict, float]],
            bm25_results: List[Tuple[Dict, float]],
            vector_scores: List[float],
            bm25_scores: List[float]
    ) -> Dict[str, Dict]:
        """Combine vector and BM25 scores with alpha weighting"""
        combined_scores = {}

        # Add vector scores
        for (chunk, _), score in zip(vector_results, vector_scores):
            chunk_id = chunk['chunk_id']
            combined_scores[chunk_id] = {
                'chunk': chunk,
                'score': self.alpha * score
            }

        # Add BM25 scores
        for (chunk, _), score in zip(bm25_results, bm25_scores):
            chunk_id = chunk['chunk_id']
            weighted_score = (1 - self.alpha) * score

            if chunk_id in combined_scores:
                combined_scores[chunk_id]['score'] += weighted_score
            else:
                combined_scores[chunk_id] = {'chunk': chunk, 'score': weighted_score}

        return combined_scores

    def _rerank_results(
            self,
            query: str,
            results: List[Tuple[Dict, float]],
            k: int
    ) -> List[Tuple[Dict, float]]:
        """Apply cross-encoder reranking to results"""
        if not results:
            return results

        try:
            candidates = results[:min(len(results), k * 2)]
            pairs = [[query, chunk['text']] for chunk, _ in candidates]

            @rerank_circuit_breaker
            def rerank_with_protection():
                return self.reranker.predict(pairs)

            raw_scores = rerank_with_protection()
            scores = self._normalize_scores_pytorch(list(raw_scores))

            reranked = [
                (candidates[i][0], float(scores[i]))
                for i in range(len(candidates))
            ]
            reranked.sort(key=lambda x: x[1], reverse=True)

            return reranked[:k]

        except Exception as e:
            logger.warning("Reranking failed, falling back to hybrid results: {e}")
            return results[:k]

    def _normalize_scores_pytorch(self, scores: List[float]) -> List[float]:
        """Normalize using z-score + sigmoid to 0-1 range"""
        import math

        if not scores:
            return []

        # Calculate mean and std
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        std = variance ** 0.5

        if std == 0:
            return [0.5] * len(scores)

        # Z-score + sigmoid
        z_scores = [(score - mean) / std for score in scores]

        return [1 / (1 + math.exp(-z)) for z in z_scores]

    def get_system_info(self):
        vector_info = self.vector_retriever.get_device_info()
        cache_stats = cache.get_stats()

        return {
            **vector_info,
            "hybrid_alpha": self.alpha,
            "total_chunks": len(self.vector_retriever.chunks) if self.vector_retriever.chunks else 0,
            "cache_stats": cache_stats,
            "circuit_breaker_enabled": True
        }
