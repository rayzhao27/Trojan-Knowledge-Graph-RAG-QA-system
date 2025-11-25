import re
import torch
import logging
import numpy as np

from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        logger.info("Using rank_bm25 library")

    def build_index(self, chunks: List[Dict]):
        """Build BM25 index from chunks"""
        self.chunks = chunks
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")

        tokenized_corpus = [self._tokenize(chunk['text']) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"Built BM25 index with {len(chunks)} chunks")

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for relevant chunks using BM25"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Use PyTorch for faster topk on GPU if available
        if torch.cuda.is_available():
            scores_tensor = torch.tensor(scores, device='cuda')
            top_indices = torch.topk(scores_tensor, min(k, len(scores))).indices.cpu().numpy()
        else:
            top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only positive scores
                results.append((self.chunks[idx], float(scores[idx])))

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        return [token for token in tokens if token not in stop_words and len(token) > 2]
