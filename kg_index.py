from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING
from dataclasses import dataclass
from sklearn.cluster import KMeans
import faiss
faiss.omp_set_num_threads(1)
import logging
from collections import defaultdict

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from kg_forest_new import Entity, Relation, ChunkExtraction

@dataclass
class IndexConfig:
    """索引配置"""
    n_bits: int = 8      # 量化位数
    n_subquantizers: int = 8  # 子量化器数量

class MultiLevelIndex:
    """多级索引类"""
    
    def __init__(self, config: IndexConfig = IndexConfig()):
        """
        初始化多级索引
        
        Args:
            config: 索引配置参数
        """
        self.config = config
        
        # ID管理
        self.entity_id_counter = 0
        self.relation_id_counter = 0
        self.entity_name_to_id: Dict[str, int] = {}
        self.relation_name_to_id: Dict[str, int] = {}
        self.entity_id_to_name: Dict[int, str] = {}
        
        # 向量存储
        self.entity_vectors: List[np.ndarray] = []
        self.relation_vectors: List[np.ndarray] = []
        
        # 关系索引 - 简化版
        self.entity_to_relations: Dict[int, Set[int]] = defaultdict(set)
        
        # Chunk索引
        self.entity_relation_to_chunks: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        
        # FAISS索引
        self.entity_index = None
        
    def get_or_create_entity_id(self, entity_name: str) -> Tuple[int, bool]:
        """
        获取或创建实体ID
        
        Args:
            entity_name: 实体名称
        Returns:
            Tuple[int, bool]: (实体ID, 是否为新创建的实体)
        """
        if entity_name not in self.entity_name_to_id:
            self.entity_name_to_id[entity_name] = self.entity_id_counter
            self.entity_id_to_name[self.entity_id_counter] = entity_name
            self.entity_id_counter += 1
            logging.info(f"Created new entity: {entity_name} (ID: {self.entity_id_counter-1})")
            return self.entity_name_to_id[entity_name], True
        logging.debug(f"Found existing entity: {entity_name} (ID: {self.entity_name_to_id[entity_name]})")
        return self.entity_name_to_id[entity_name], False
    
    def get_or_create_relation_id(self, relation_name: str) -> Tuple[int, bool]:
        """
        获取或创建关系ID
        
        Args:
            relation_name: 关系名称
        Returns:
            Tuple[int, bool]: (关系ID, 是否为新创建的关系)
        """
        if relation_name not in self.relation_name_to_id:
            self.relation_name_to_id[relation_name] = self.relation_id_counter
            self.relation_id_counter += 1
            logging.info(f"Created new relation: {relation_name} (ID: {self.relation_id_counter-1})")
            return self.relation_name_to_id[relation_name], True
        logging.debug(f"Found existing relation: {relation_name} (ID: {self.relation_name_to_id[relation_name]})")
        return self.relation_name_to_id[relation_name], False
    
    def add_entity_vector(self, entity_id: int, vector: np.ndarray):
        """添加实体向量"""
        while len(self.entity_vectors) <= entity_id:
            self.entity_vectors.append(np.zeros(vector.shape[0], dtype=np.float32))
        self.entity_vectors[entity_id] = vector
    
    def add_relation_vector(self, relation_id: int, vector: np.ndarray):
        """添加关系向量"""
        while len(self.relation_vectors) <= relation_id:
            self.relation_vectors.append(np.zeros(vector.shape[0], dtype=np.float32))
        self.relation_vectors[relation_id] = vector
    
    def add_entity_relation(self, entity_id: int, relation_id: int):
        """添加实体-关系关联"""
        self.entity_to_relations[entity_id].add(relation_id)
    
    def add_chunk_reference(self, entity_id: int, relation_id: int, chunk_id: str):
        """添加实体-关系到chunk的引用"""
        self.entity_relation_to_chunks[(entity_id, relation_id)].add(chunk_id)
    
    def build_faiss_indices(self):
        """构建FAISS索引"""
        # 构建实体索引
        if len(self.entity_vectors) == 0:
            logging.warning("No entity vectors to build index")
            return
            
        vectors = np.array(self.entity_vectors)
        dimension = vectors.shape[1]
        
        # 根据向量数量动态选择索引类型
        if len(vectors) < 1000:  # 数据量小时使用简单索引
            logging.info("Using simple FlatL2 index for small dataset")
            self.entity_index = faiss.IndexFlatL2(dimension)
        else:  # 数据量大时使用量化索引
            logging.info("Using IVFPQ index for large dataset")
            # 动态计算聚类数量，确保每个聚类至少有39个样本
            n_clusters = max(1, min(len(vectors) // 39, 100))  # 最多100个聚类
            # 动态调整子量化器数量，确保不超过向量维度
            n_subquantizers = min(self.config.n_subquantizers, dimension)
            
            logging.info(f"Creating IVFPQ index with {n_clusters} clusters and {n_subquantizers} subquantizers")
            quantizer = faiss.IndexFlatL2(dimension)
            self.entity_index = faiss.IndexIVFPQ(
                quantizer, dimension, n_clusters,
                n_subquantizers, self.config.n_bits
            )
            # 训练并添加向量
            self.entity_index.train(vectors)
        
        # 添加向量到索引
        self.entity_index.add(vectors)
        logging.info(f"Built FAISS index for {len(vectors)} entity vectors")
    
    def process_chunk_extraction(self, chunk_id: str, extraction: "ChunkExtraction", 
                                 entity_embedder=None, relation_embedder=None):
        """
        处理ChunkExtraction对象，提取实体和关系并添加到索引中
        
        Args:
            chunk_id: 文本块ID
            extraction: ChunkExtraction对象
            entity_embedder: 用于生成实体向量的嵌入模型
            relation_embedder: 用于生成关系向量的嵌入模型
        """
        logging.info(f"\nProcessing chunk {chunk_id}:")
        logging.info(f"Found {len(extraction.entities)} entities and {len(extraction.relations)} relations")
        
        # 记录所有在关系中出现的实体
        relation_entities: Set[int] = set()
        
        # 处理所有关系
        for relation in extraction.relations:
            # 确保关系中的实体都存在
            source_id, is_new_source = self.get_or_create_entity_id(relation.source_entity)
            target_id, is_new_target = self.get_or_create_entity_id(relation.target_entity)
            relation_id, is_new_relation = self.get_or_create_relation_id(relation.relation)
            
            logging.info(f"Processing relation: {relation.source_entity} --[{relation.relation}]--> {relation.target_entity}")
            
            # 添加实体-关系关联
            self.add_entity_relation(source_id, relation_id)
            self.add_entity_relation(target_id, relation_id)
            
            # 标记在关系中出现的实体
            relation_entities.add(source_id)
            relation_entities.add(target_id)
            
            # 如果提供了关系嵌入模型，且是新关系，生成关系向量
            if relation_embedder is not None and is_new_relation:
                relation_vector = relation_embedder.encode(
                    f"{relation.relation}",
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype(np.float32)
                self.add_relation_vector(relation_id, relation_vector)
            
            # 如果提供了实体嵌入模型，且是新实体，分别生成实体向量
            if entity_embedder is not None:
                if is_new_source:
                    source_vector = entity_embedder.encode(
                        relation.source_entity,
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    ).astype(np.float32)
                    self.add_entity_vector(source_id, source_vector)
                
                if is_new_target:
                    target_vector = entity_embedder.encode(
                        relation.target_entity,
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    ).astype(np.float32)
                    self.add_entity_vector(target_id, target_vector)
            
            # 添加关系-实体到chunk的引用
            self.add_chunk_reference(source_id, relation_id, chunk_id)
            self.add_chunk_reference(target_id, relation_id, chunk_id)
        
        # 处理所有实体：仅对未在关系中出现的实体添加引用
        for entity in extraction.entities:
            entity_id, is_new = self.get_or_create_entity_id(entity.entity_name)
            if entity_id in relation_entities:
                logging.debug(f"Skipping entity {entity.entity_name} as it's already processed in relations")
                continue
                
            logging.info(f"Processing standalone entity: {entity.entity_name}")
            
            # 如果提供了实体嵌入模型，且是新实体，生成实体向量
            if entity_embedder is not None and is_new:
                entity_vector = entity_embedder.encode(
                    entity.entity_name,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype(np.float32)
                self.add_entity_vector(entity_id, entity_vector)
            
            # 添加实体到chunk的引用（无关系时 relation_id=-1）
            self.add_chunk_reference(entity_id, -1, chunk_id)
        
        logging.info(f"Finished processing chunk {chunk_id}")
        logging.info(f"Current total entities: {len(self.entity_name_to_id)}")
        logging.info(f"Current total relations: {len(self.relation_name_to_id)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, List[str]]]:
        """
        搜索知识图谱
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
        Returns:
            List[Tuple[str, List[str]]]: [(实体名称, [相关chunk_id列表]), ...]
        """
        if self.entity_index is None:
            logging.warning("Entity index not built")
            return []
            
        # 确保查询向量是2D数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # 搜索最相似的实体
        distances, indices = self.entity_index.search(query_vector, top_k)
        
        # 收集结果
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # 跳过无效结果
                continue
                
            entity_id = int(idx)
            if entity_id not in self.entity_id_to_name:
                continue
                
            entity_name = self.entity_id_to_name[entity_id]
            chunk_ids = set()
            
            # 获取实体相关的所有关系
            relation_ids = self.entity_to_relations.get(entity_id, set())
            if relation_ids:
                # 如果有关系，获取所有相关的chunk
                for relation_id in relation_ids:
                    chunk_refs = self.entity_relation_to_chunks.get((entity_id, relation_id), set())
                    chunk_ids.update(chunk_refs)
            else:
                # 如果没有关系，获取独立实体的chunk
                chunk_refs = self.entity_relation_to_chunks.get((entity_id, -1), set())
                chunk_ids.update(chunk_refs)
                
            if chunk_ids:  # 只返回有chunk引用的实体
                results.append((entity_name, list(chunk_ids)))
                
        return results

    def search_by_name(self, entity_embedding: np.ndarray, relation_embedding: Optional[np.ndarray] = None, top_k: int = 5) -> List[Tuple[str, List[str]]]:
        """
        通过实体和关系的嵌入向量搜索，使用FAISS索引进行相似度匹配
        
        Args:
            entity_embedding: 实体嵌入向量
            relation_embedding: 关系嵌入向量（可选）
            top_k: 返回结果数量
        Returns:
            List[Tuple[str, List[str]]]: [(实体名称, [相关chunk_id列表]), ...]
        """
        # 确保查询向量是2D数组
        if entity_embedding.ndim == 1:
            entity_embedding = entity_embedding.reshape(1, -1)
            
        # 使用FAISS搜索相似实体
        distances, indices = self.entity_index.search(entity_embedding, top_k)
        
        # 收集结果
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # 跳过无效结果
                continue
                
            similar_entity_id = int(idx)
            if similar_entity_id not in self.entity_id_to_name:
                continue
                
            similar_entity_name = self.entity_id_to_name[similar_entity_id]
            chunk_ids = set()
            
            if relation_embedding is None:
                # 如果没有指定关系向量，获取所有相关的chunk
                # 1. 获取有关系的chunk
                for rel_id in self.entity_to_relations.get(similar_entity_id, set()):
                    chunk_refs = self.entity_relation_to_chunks.get((similar_entity_id, rel_id), set())
                    chunk_ids.update(chunk_refs)
                # 2. 获取独立实体的chunk
                chunk_refs = self.entity_relation_to_chunks.get((similar_entity_id, -1), set())
                chunk_ids.update(chunk_refs)
            else:
                # 如果指定了关系向量，计算相似度并只返回最相关的关系的chunk
                if similar_entity_id not in self.entity_to_relations:
                    continue
                    
                # 计算与所有关系的相似度
                relation_similarities = []
                for rel_id in self.entity_to_relations[similar_entity_id]:
                    if rel_id < len(self.relation_vectors):
                        rel_vector = self.relation_vectors[rel_id]
                        similarity = np.dot(relation_embedding, rel_vector) / (
                            np.linalg.norm(relation_embedding) * np.linalg.norm(rel_vector)
                        )
                        relation_similarities.append((rel_id, similarity))
                
                if relation_similarities:
                    # 获取最相关的关系
                    best_rel_id = max(relation_similarities, key=lambda x: x[1])[0]
                    chunk_refs = self.entity_relation_to_chunks.get((similar_entity_id, best_rel_id), set())
                    chunk_ids.update(chunk_refs)
                
            if chunk_ids:  # 只返回有chunk引用的实体
                results.append((similar_entity_name, list(chunk_ids)))
                
        return results
