import logging
import numpy as np

from typing import List, Dict, Tuple, Set

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.graph = {}
        self.chunks = {}
        
    def build_graph(self, chunks: List[Dict], embeddings: np.ndarray):
        """Build graph by connecting similar chunks"""
        logger.info(f"Building knowledge graph with {len(chunks)} nodes...")
        
        # Store chunks
        for chunk in chunks:
            self.chunks[chunk['chunk_id']] = chunk
            self.graph[chunk['chunk_id']] = []
        
        # Connect similar chunks
        for i, chunk_i in enumerate(chunks):
            for j, chunk_j in enumerate(chunks):
                # Skip self-loop and duplicates
                if i >= j:
                    continue
                
                # Calculate similarity
                similarity = np.dot(embeddings[i], embeddings[j])
                
                if similarity > self.similarity_threshold:
                    # Bidirectional connection
                    self.graph[chunk_i['chunk_id']].append(chunk_j['chunk_id'])
                    self.graph[chunk_j['chunk_id']].append(chunk_i['chunk_id'])
        
        # Count connections
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        logger.info(f"Graph built with {total_edges} connections")
    
    def expand_results(
        self, 
        initial_results: List[Tuple[Dict, float]], 
        max_hops: int = 1,
        max_total: int = 10
    ) -> List[Tuple[Dict, float]]:
        """Expand search results by traversing the graph"""
        
        if not self.graph:
            return initial_results
        
        expanded = {}
        visited = set()
        
        # Add initial results
        for chunk, score in initial_results:
            chunk_id = chunk['chunk_id']
            expanded[chunk_id] = (chunk, score)
            visited.add(chunk_id)
        
        # Traverse graph
        for hop in range(max_hops):
            new_chunks = []
            
            for chunk_id in list(visited):
                if chunk_id not in self.graph:
                    continue
                
                # Get neighbors
                for neighbor_id in self.graph[chunk_id]:
                    if neighbor_id not in visited:
                        neighbor_chunk = self.chunks[neighbor_id]
                        # Decay score based on hop distance
                        decay_factor = 0.5 ** (hop + 1)
                        neighbor_score = expanded[chunk_id][1] * decay_factor
                        new_chunks.append((neighbor_id, neighbor_chunk, neighbor_score))
                        visited.add(neighbor_id)
            
            # Add new chunks
            for neighbor_id, neighbor_chunk, neighbor_score in new_chunks:
                if neighbor_id not in expanded:
                    expanded[neighbor_id] = (neighbor_chunk, neighbor_score)
        
        # Sort by score and limit
        results = sorted(expanded.values(), key=lambda x: x[1], reverse=True)

        return results[:max_total]
    
    def get_context_window(self, chunk_id: str, window_size: int = 2) -> List[Dict]:
        """Get surrounding chunks from the same document"""
        if chunk_id not in self.chunks:
            return []
        
        chunk = self.chunks[chunk_id]
        source = chunk['source']
        page = chunk['page_num']
        
        # Find chunks from the same document and nearby pages
        context = []
        for cid, c in self.chunks.items():
            if c['source'] == source:
                page_diff = abs(c['page_num'] - page)
                if page_diff <= window_size:
                    context.append(c)
        
        # Sort by page number
        context.sort(key=lambda x: x['page_num'])
        return context
    
    def get_stats(self) -> Dict:
        if not self.graph:
            return {"nodes": 0, "edges": 0, "avg_connections": 0}
        
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        avg_connections = sum(len(neighbors) for neighbors in self.graph.values()) / len(self.graph)
        
        return {
            "nodes": len(self.graph),
            "edges": total_edges,
            "avg_connections": round(avg_connections, 2)
        }
