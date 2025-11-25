import logging
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import faiss
import os
import numpy as np

logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Force PyTorch backend
        os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load model with device specification
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.chunks = []

        logger.info(f"PyTorch Based VectorRetriever initialized with {model_name}")

    def build_index(self, chunks: List[Dict], index_file: str = None):
        """Build FAISS index from chunks using PyTorch"""
        self.chunks = chunks
        
        # Try to load existing index
        if index_file and os.path.exists(index_file):
            logger.info(f"Loading FAISS index from {index_file}")
            self.index = faiss.read_index(index_file)
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return
        
        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Encoding {len(texts)} texts with PyTorch backend...")

        try:
            # Generate embeddings with PyTorch backend and memory management
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    device=self.device,
                    batch_size=32,
                    num_workers=0
                )

            # Ensure embeddings are numpy array
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Ensure float32 type and contiguous
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

            # Manual normalization to avoid FAISS issues
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / norms
            embeddings_normalized = embeddings_normalized.astype(np.float32)

            self.index.add(embeddings_normalized)

            logger.info(f"Built PyTorch-powered index with {self.index.ntotal} vectors")
            
            # Save index if path provided
            if index_file:
                faiss.write_index(self.index, index_file)
                logger.info(f"Saved FAISS index to {index_file}")

        except Exception as e:
            print(f"[ERROR] Error building index: {e}")
            logger.info(f"Embeddings info: type={type(embeddings)}, shape={getattr(embeddings, 'shape', 'no shape')}")
            raise

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search using PyTorch backend"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query with memory management
        with torch.no_grad():
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                device=self.device,
                num_workers=0
            )

        # Ensure query embedding is numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        # Ensure float32 type and make a copy
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32).copy()

        # Normalize and search
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):  # Safety check
                results.append((self.chunks[idx], float(score)))

        return results

    def get_device_info(self):
        """Get information about the current device"""
        return {
            "device": self.device,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
