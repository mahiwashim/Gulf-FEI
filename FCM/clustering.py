"""
Fuzzy C-Means Clustering Module
Clusters RAG results based on semantic similarity and a fixed relation threshold.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


class FCMClustering:
    """Fuzzy C-Means style clustering with relation-map generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize FCM Clustering
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.data = []
        self.clusters = {}
        self.centroids = None
    
    def cluster_rag_results(
        self,
        rag_results: List[Tuple[str, str, float]],
        num_clusters: Optional[int] = None,
        relation_threshold: float = 0.7,
        max_relation_lines: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Cluster RAG results based on semantic similarity
        
        Args:
            rag_results: List of (text, source, score) from RAG
            num_clusters: Number of clusters (auto if None)
            relation_threshold: Threshold for relationship strength (0-1)
            
        Returns:
            Clustering results with cluster assignments and distances
        """
        if not rag_results:
            return {
                "clusters": {},
                "cluster_distribution": {},
                "num_clusters": 0,
                "relation_threshold": relation_threshold,
                "total_members": 0,
                "total_relation_lines": 0,
                "relation_edges": [],
                "fcm_map": {"nodes": [], "edges": [], "relation_threshold": relation_threshold}
            }
        
        # Extract texts and embeddings
        texts = [text for text, _, _ in rag_results]
        self.data = rag_results
        self.embeddings = self.model.encode(texts)
        
        # Determine optimal number of clusters
        if num_clusters is None:
            num_clusters = min(3, max(1, len(rag_results) // 2))
        num_clusters = min(num_clusters, len(rag_results))
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        self.centroids = kmeans.cluster_centers_
        
        # Calculate distances to centroids
        distances = kmeans.transform(self.embeddings)
        max_distance = np.max(distances)
        
        # Organize clusters
        clusters = {}
        cluster_distribution = {}
        
        for cluster_id in range(num_clusters):
            cluster_members = []
            member_indices = np.where(labels == cluster_id)[0]
            
            for idx in member_indices:
                text, source, rag_score = rag_results[idx]
                
                # Calculate relationship strength based on distance
                min_dist_to_centroid = distances[idx, cluster_id]
                # Convert distance to relationship score (0-1, closer = stronger)
                relationship_score = 1 - (min_dist_to_centroid / (max_distance + 1e-6))
                
                # Apply threshold filter
                if relationship_score >= relation_threshold:
                    cluster_members.append({
                        "member_id": f"m_{idx}",
                        "embedding_index": int(idx),
                        "text": text,
                        "source": source,
                        "rag_score": float(rag_score),
                        "relationship_score": float(relationship_score),
                        "distance_to_centroid": float(min_dist_to_centroid)
                    })
            
            if cluster_members:
                # Sort by relationship score
                cluster_members.sort(key=lambda x: x["relationship_score"], reverse=True)
                clusters[f"cluster_{cluster_id}"] = {
                    "members": cluster_members,
                    "size": len(cluster_members),
                    "relation_lines": 0,
                    "primary_topic": self._extract_topic(cluster_members[0]["text"])
                }
                cluster_distribution[f"cluster_{cluster_id}"] = len(cluster_members)

        relation_edges = self._build_relation_edges(clusters, relation_threshold)
        if max_relation_lines is not None and max_relation_lines > 0:
            relation_edges = sorted(
                relation_edges,
                key=lambda edge: edge["weight"],
                reverse=True
            )[:max_relation_lines]
        total_relation_lines = len(relation_edges)
        edge_count_by_cluster: Dict[str, int] = {}
        for edge in relation_edges:
            cname = edge["cluster"]
            edge_count_by_cluster[cname] = edge_count_by_cluster.get(cname, 0) + 1
        for cname, cluster in clusters.items():
            cluster["relation_lines"] = edge_count_by_cluster.get(cname, 0)
        fcm_map = self._build_fcm_map(clusters, relation_edges, relation_threshold)

        return {
            "clusters": clusters,
            "cluster_distribution": cluster_distribution,
            "num_clusters": num_clusters,
            "relation_threshold": relation_threshold,
            "total_members": sum(cluster_distribution.values()),
            "total_relation_lines": total_relation_lines,
            "relation_edges": relation_edges,
            "fcm_map": fcm_map
        }
    
    def _extract_topic(self, text: str, max_length: int = 50) -> str:
        """Extract topic from text"""
        # Simple topic extraction - first sentence or up to max_length
        sentence = text.split('.')[0].strip()
        if len(sentence) > max_length:
            sentence = sentence[:max_length] + "..."
        return sentence
    
    def calculate_cluster_proximity(
        self,
        cluster_results: Dict
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Calculate proximity between clusters
        
        Returns proximity matrix-like structure
        """
        if self.centroids is None or self.centroids.shape[0] < 2:
            return {}
        
        proximity = {}
        num_clusters = self.centroids.shape[0]
        
        for i in range(num_clusters):
            proximity[f"cluster_{i}"] = {}
            for j in range(num_clusters):
                if i != j:
                    # Calculate distance between centroids
                    dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    # Normalize to 0-1 range (closer = higher value)
                    max_possible_dist = 2 * np.linalg.norm(self.centroids[i])
                    proximity_score = 1 - (dist / (max_possible_dist + 1e-6))
                    proximity[f"cluster_{i}"][f"cluster_{j}"] = float(proximity_score)
        
        return proximity
    
    def get_visualization_data(self, cluster_results: Dict) -> Dict:
        """
        Prepare data for visualization
        
        Returns data suitable for plotly/charts
        """
        if self.embeddings is None:
            return {}

        if isinstance(self.embeddings, np.ndarray) and self.embeddings.shape[0] == 0:
            return {}
        
        # Convert list to numpy array if needed
        if isinstance(self.embeddings, list):
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
        else:
            embeddings_array = np.asarray(self.embeddings, dtype=np.float32)
        
        # Use first 2 PCA components for 2D visualization
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_array)
        except:
            # Fallback if PCA fails
            embeddings_2d = embeddings_array[:, :2] if embeddings_array.shape[1] >= 2 else embeddings_array
        
        # Prepare visualization data
        viz_data = {
            "points": [],
            "edges": [],
            "clusters": cluster_results.get("cluster_distribution", {}),
            "explained_variance": None
        }
        
        cluster_colors = {
            "cluster_0": "#056F73",  # Gulf teal
            "cluster_1": "#27A798",  # Data current
            "cluster_2": "#013056",  # Gulf authority
            "cluster_3": "#F4F7F6"   # Tidal mist
        }
        
        for i, (text, source, score) in enumerate(self.data):
            # Find which cluster this belongs to
            cluster_name = None
            for cname, cluster_data in cluster_results.get("clusters", {}).items():
                for member in cluster_data.get("members", []):
                    if member["source"] == source and member["text"] == text:
                        cluster_name = cname
                        break
                if cluster_name:
                    break
            
            if cluster_name:
                viz_data["points"].append({
                    "id": f"m_{i}",
                    "x": float(embeddings_2d[i, 0]),
                    "y": float(embeddings_2d[i, 1]),
                    "text": text[:100] + "...",
                    "source": source,
                    "cluster": cluster_name,
                    "color": cluster_colors.get(cluster_name, "#056F73"),
                    "relationship_score": float(score)
                })

        point_map = {point["id"]: point for point in viz_data["points"]}
        for edge in cluster_results.get("relation_edges", []):
            source_point = point_map.get(edge["source_id"])
            target_point = point_map.get(edge["target_id"])
            if not source_point or not target_point:
                continue

            viz_data["edges"].append({
                "x0": source_point["x"],
                "y0": source_point["y"],
                "x1": target_point["x"],
                "y1": target_point["y"],
                "weight": float(edge["weight"]),
                "source": edge["source"],
                "target": edge["target"],
                "cluster": edge["cluster"]
            })

        return viz_data

    def _build_relation_edges(self, clusters: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
        """Create weighted relation edges between members in each cluster."""
        if self.embeddings is None:
            return []

        embedding_matrix = np.asarray(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(embedding_matrix, axis=1) + 1e-12
        normalized = embedding_matrix / norms[:, None]

        relation_edges: List[Dict[str, Any]] = []
        for cluster_name, cluster in clusters.items():
            members = cluster.get("members", [])
            if len(members) < 2:
                continue

            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a = members[i]
                    b = members[j]
                    idx_a = a["embedding_index"]
                    idx_b = b["embedding_index"]
                    similarity = float(np.dot(normalized[idx_a], normalized[idx_b]))

                    if similarity >= threshold:
                        relation_edges.append({
                            "source_id": a["member_id"],
                            "target_id": b["member_id"],
                            "source": a["source"],
                            "target": b["source"],
                            "weight": similarity,
                            "cluster": cluster_name
                        })

        return relation_edges

    def _build_fcm_map(
        self,
        clusters: Dict[str, Any],
        edges: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, Any]:
        """Build a frontend-ready map structure with nodes and weighted edges."""
        nodes: List[Dict[str, Any]] = []
        for cluster_name, cluster in clusters.items():
            for member in cluster.get("members", []):
                nodes.append({
                    "id": member["member_id"],
                    "label": member["source"],
                    "cluster": cluster_name,
                    "rag_score": member["rag_score"],
                    "relationship_score": member["relationship_score"]
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "relation_threshold": threshold
        }
