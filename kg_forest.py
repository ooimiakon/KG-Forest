import os
import json
import re
import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
import openai
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from pathlib import Path
import torch
import shutil
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

class Entity:
    def __init__(self, name: str, type: str, description: str):
        self.name = name
        self.type = type
        self.description = description
        self.id = str(uuid.uuid4())

class Relation:
    def __init__(self, source: Entity, relation: str, target: Entity, description: str):
        self.source = source
        self.relation = relation
        self.target = target
        self.description = description
        self.id = str(uuid.uuid4())

class KnowledgeGraph:
    def __init__(self):
        self.entities = {}  # entity_id -> Entity
        self.relations = {}  # relation_id -> Relation
        self.entity_name_to_id = {}  # entity_name -> entity_id
        self.relation_triples = set()  # (source_id, relation, target_id)

    def lookup_entity_attribute(self, entity_name: str, attribute_name: Optional[str] = None) -> Any:
        """根据实体名称和属性名称查找实体的属性值"""
        entity_id = self.entity_name_to_id.get(entity_name)
        if entity_id is None:
            return "Entity not found."
        
        entity = self.entities.get(entity_id)
        if entity is None:
            return "Entity not found."
        
        attributes = {
            "name": entity.name,
            "type": entity.type,
            "description": entity.description
        }
        
        if attribute_name is None:
            return attributes
        else:
            value = attributes.get(attribute_name)
            if value is not None:
                return {attribute_name: value}
            else:
                return {f"Attribute '{attribute_name}' not found in '{entity.name}'": attributes}
    
    def lookup_entity_relation(self, entity_name: str, relation_name: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Lookup related target entities given an entity and optional relation.
        
        Returns a list of (relation, target_entity_name) pairs:
        - If relation_name is given, returns only matching relations.
        - If relation_name is None, returns all relations under the entity.
        - If entity not found, returns an empty list.
        """
        results = []
        entity_id = self.entity_name_to_id.get(entity_name)
        if entity_id is None:
            print(f"Entity '{entity_name}' not found.")
            return results
        
        for src_id, relation, tgt_id in self.relation_triples:
            if src_id == entity_id:
                target_entity = self.entities.get(tgt_id)
                if target_entity:
                    results.append(("outgoing", relation, target_entity.name))
            elif tgt_id == entity_id:
                source_entity = self.entities.get(src_id)
                if source_entity:
                    results.append(("incoming", relation, source_entity.name))
        
        if not results:
            print(f"No relations found for entity '{entity_name}'.")
        
        return results
    def print_entity_relations(self):
        """Print all relations for each entity: outgoing and incoming."""
        for eid, entity in self.entities.items():
            print(f"\nEntity: {entity.name} (ID: {eid})")

            outgoing = []
            incoming = []

            for src_id, relation, tgt_id in self.relation_triples:
                if src_id == eid:
                    target_entity = self.entities.get(tgt_id)
                    if target_entity:
                        outgoing.append((relation, target_entity.name))
                elif tgt_id == eid:
                    source_entity = self.entities.get(src_id)
                    if source_entity:
                        incoming.append((relation, source_entity.name))
            
            if outgoing:
                print("  Outgoing Relations:")
                for relation, target in outgoing:
                    print(f"    {relation} -> {target}")
            else:
                print("  Outgoing Relations: None")

            if incoming:
                print("  Incoming Relations:")
                for relation, source in incoming:
                    print(f"    {relation} <- {source}")
            else:
                print("  Incoming Relations: None")


class TextProcessor:
    def __init__(self, chunk_size: int = 600):
        self.chunk_size = chunk_size
        self.punct_re = re.compile(r'[\\.!?]')

    def clean_text(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_into_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        text = text.strip()
        spans = []
        i, L = 0, len(text)
        while i < L:
            end = min(L, i + self.chunk_size)
            m = self.punct_re.search(text, end)
            if m:
                end = m.end()
            chunk = text[i:end].strip()
            spans.append((chunk, i, end))
            i = end
        logging.info(f"Created {len(spans)} chunks from text (max_chars={self.chunk_size})")
        return spans

class RelationExtractor:
    def __init__(self, api_key: str, base_url: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_entities_relations",
                    "description": "Extract entities and relations from a text chunk, with self-audit in one pass",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "entity_name": {"type": "string"},
                                        "entity_type": {"type": "string"},
                                        "entity_description": {"type": "string"},
                                    },
                                    "required": ["entity_name", "entity_type", "entity_description"]
                                }
                            },
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_entity": {"type": "string"},
                                        "relation": {"type": "string"},
                                        "target_entity": {"type": "string"},
                                        "relation_description": {"type": "string"},
                                    },
                                    "required": ["source_entity", "relation", "target_entity", "relation_description"]
                                }
                            }
                        },
                        "required": ["entities", "relations"]
                    }
                }
            }
        ]
        self.few_shot = [
            {
                "role": "system",
                "content": (
                    "You are an information extraction assistant.  \n"
                    "Perform these steps in one pass:\n"
                    "1) Extract entities with {entity_name, entity_type, entity_description};\n"
                    "2) Extract relations as triples {source_entity, relation, target_entity, relation_description};\n"
                    "3) Self-audit: review your own output and add any missing items.\n"
                    "Return exactly a JSON matching the tool schema."
                    "-Always output at least one relation for the main action or verb implied by the question\n"
                    "Example:\n"
                        "Input: \"Who works for Acme Corp?\n"
                        "Entities: \n"
                            "  - { entity_name: \"Acme Corp\", entity_type: \"Organization\", entity_description: \"Company mentioned in query\" }\n"
                        "Relations:\n"
                            "  - { source_entity: \"Unknown person\", relation: \"works_for\", target_entity: \"Acme Corp\", relation_description: \"Person is employed by Acme Corp\" }\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Chunk:\n"
                    "\"Alice joined Acme Corp in 2021 and moved to London for work.\""
                )
            },
            {
                "role": "assistant",
                "content": "I will extract entities and relations from this text.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "extract_entities_relations",
                            "arguments": json.dumps({
                                "entities": [
                                    {"entity_name": "Alice", "entity_type": "Person", "entity_description": "An individual who joined a company in 2021"},
                                    {"entity_name": "Acme Corp", "entity_type": "Organization", "entity_description": "Company Alice joined in 2021"},
                                    {"entity_name": "London", "entity_type": "Location", "entity_description": "City Alice moved to for work"}
                                ],
                                "relations": [
                                    {"source_entity": "Alice", "relation": "employment", "target_entity": "Acme Corp", "relation_description": "Alice joined Acme Corp in 2021"},
                                    {"source_entity": "Alice", "relation": "relocation", "target_entity": "London", "relation_description": "Alice moved to London for work"}
                                ]
                            })
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Successfully extracted entities and relations."
            }
        ]

    def extract_from_chunk(self, chunk: str) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.few_shot + [{"role":"user","content":f"Chunk:\n\"{chunk}\""}],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "extract_entities_relations"}},
                temperature=0.0
            )
            logging.info(f"Response content: {resp}")
            if resp.choices[0].message.tool_calls:
                tool_call = resp.choices[0].message.tool_calls[0]
                return json.loads(tool_call.function.arguments)
            else:
                logging.error("No tool calls in response")
                raise Exception("No tool calls in response")
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            raise

class GraphStorage:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.ensure_storage_dir()

    def ensure_storage_dir(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "chunks").mkdir(exist_ok=True)
        (self.storage_path / "embeddings").mkdir(exist_ok=True)

    def save_graph(self, kg: KnowledgeGraph):
        graph_data = {
            "entities": {eid: vars(entity) for eid, entity in kg.entities.items()},
            "relations": {rid: {
                "source": relation.source.id,
                "relation": relation.relation,
                "target": relation.target.id,
                "description": relation.description
            } for rid, relation in kg.relations.items()}
        }
        with open(self.storage_path / "graph.json", "w") as f:
            json.dump(graph_data, f, indent=2)

    def load_graph(self) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        try:
            with open(self.storage_path / "graph.json", "r") as f:
                graph_data = json.load(f)
            
            # 重建实体
            for eid, entity_data in graph_data["entities"].items():
                entity = Entity(
                    name=entity_data["name"],
                    type=entity_data["type"],
                    description=entity_data["description"]
                )
                entity.id = eid  # 保持原有ID
                kg.entities[eid] = entity
                kg.entity_name_to_id[entity.name] = eid
            
            # 重建关系
            for rid, relation_data in graph_data["relations"].items():
                source = kg.entities[relation_data["source"]]
                target = kg.entities[relation_data["target"]]
                relation = Relation(
                    source=source,
                    relation=relation_data["relation"],
                    target=target,
                    description=relation_data["description"]
                )
                relation.id = rid  # 保持原有ID
                kg.relations[rid] = relation
                kg.relation_triples.add((source.id, relation.relation, target.id))
            
            return kg
        except FileNotFoundError:
            return kg

class KGStorage:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.ensure_storage_dir()
        
        # 核心数据结构
        self.concept2cid: Dict[str, int] = {}  # 概念名 -> 概念ID
        self.attr2aid: Dict[str, int] = {}     # 关系名 -> 关系ID
        self.triples: List[Tuple[int, int, int, str]] = []  # (head_cid, aid, tail_cid, chunk_id)
        
        # 向量存储
        self.concept_vecs: Optional[torch.Tensor] = None  # 概念向量 (Q8)
        self.attr_vecs: Optional[torch.Tensor] = None    # 关系向量 (Q4)
        
        # 社区和聚类信息
        self.cid2scid: Dict[int, int] = {}     # 概念ID -> 社区ID
        self.cid2cluster: Dict[int, int] = {}  # 概念ID -> 簇ID
        self.cid_centroids: Dict[int, np.ndarray] = {}  # 概念ID -> 簇中心
        self.aid_centroids: Dict[int, np.ndarray] = {}  # 关系ID -> 簇中心
        
        # 初始化模型
        self.emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dim = self.emb_model.get_sentence_embedding_dimension()
    
    def ensure_storage_dir(self):
        """确保存储目录存在"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "leaf_buckets").mkdir(exist_ok=True)
    
    def quant_tensor(self, vec: torch.Tensor, bits: int) -> torch.Tensor:
        """量化张量"""
        vmax = vec.abs().max().item()
        scale = (2**(bits-1) - 1) / max(1e-6, vmax)
        return (vec * scale).round().to(torch.int8)
    
    def build_mappings(self, kg: KnowledgeGraph):
        """构建概念和关系的映射"""
        # 1. 构建概念映射
        for entity in kg.entities.values():
            name = entity.name.lower()
            self.concept2cid.setdefault(name, len(self.concept2cid))
        
        # 2. 构建关系映射
        for relation in kg.relations.values():
            pred = relation.relation.lower()
            self.attr2aid.setdefault(pred, len(self.attr2aid))
        
        # 3. 构建三元组
        for relation in kg.relations.values():
            src = relation.source.name.lower()
            trg = relation.target.name.lower()
            pred = relation.relation.lower()
            
            cid_h = self.concept2cid[src]
            cid_t = self.concept2cid[trg]
            aid = self.attr2aid[pred]
            
            # 使用文档ID作为chunk_id
            chunk_id = relation.source.id.split("_")[0]  # 假设ID格式为 "docid_entityid"
            self.triples.append((cid_h, aid, cid_t, chunk_id))
        
        # 4. 保存全局属性映射
        attr_dir = self.storage_path / "attributes"
        attr_dir.mkdir(exist_ok=True)
        
        with open(attr_dir / "mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.attr2aid, f, ensure_ascii=False)
        
        logging.info(f"Built mappings: {len(self.concept2cid)} concepts, {len(self.attr2aid)} relations, {len(self.triples)} triples")
    
    def compute_embeddings(self):
        """计算并量化概念和关系的嵌入向量"""
        # 1. 计算概念向量 (Q8)
        self.concept_vecs = torch.zeros((len(self.concept2cid), self.dim), dtype=torch.int8)
        for name, cid in self.concept2cid.items():
            text = f"{name} [Entity]"
            vec = self.emb_model.encode(text, normalize_embeddings=True, convert_to_tensor=True)
            self.concept_vecs[cid] = self.quant_tensor(vec.cpu(), bits=8)
        
        # 2. 计算关系向量 (Q4)并保存到全局存储
        attr_dir = self.storage_path / "attributes"
        self.attr_vecs = torch.zeros((len(self.attr2aid), self.dim), dtype=torch.int8)
        for pred, aid in self.attr2aid.items():
            vec = self.emb_model.encode(pred, normalize_embeddings=True, convert_to_tensor=True)
            self.attr_vecs[aid] = self.quant_tensor(vec.cpu(), bits=4)
        
        # 保存全局属性向量
        np.save(attr_dir / "embeddings.npy", self.attr_vecs.numpy())
    
    def detect_communities(self):
        """使用 Leiden 算法检测社区"""
        import igraph as ig
        
        # 1. 构建图
        G = ig.Graph(n=len(self.concept2cid), edges=[(h,t) for h,_,t,_ in self.triples], directed=False)
        
        # 2. 运行 Leiden 算法
        leiden = G.community_leiden(objective_function="modularity")
        
        # 3. 构建社区映射
        for comm_id, community in enumerate(leiden):
            for vertex in community:
                self.cid2scid[vertex] = comm_id
    
    def cluster_embeddings(self):
        """对概念进行聚类"""
        from sklearn.cluster import KMeans
        
        # 1. 转换向量为 numpy
        concept_np = self.concept_vecs.numpy().astype(np.float32)
        
        # 2. 对每个社区进行聚类
        for scid in set(self.cid2scid.values()):
            members = [cid for cid, sc in self.cid2scid.items() if sc == scid]
            if not members:
                continue
            
            # 概念聚类
            k = min(32, len(members))
            km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(concept_np[members])
            for idx, cid in enumerate(members):
                self.cid2cluster[cid] = km.labels_[idx]
                self.cid_centroids[cid] = km.cluster_centers_[km.labels_[idx]]
        
        logging.info(f"Clustered {len(self.cid2cluster)} concepts into {len(set(self.cid2cluster.values()))} clusters")
    
    def build_leaf_buckets(self, chunk_embeddings: Dict[str, np.ndarray]):
        """构建 leaf buckets 用于快速检索"""
        import faiss
        from collections import defaultdict
        
        # 1. 保存 chunk embeddings
        chunk_ids = list(chunk_embeddings.keys())
        chunk_vecs = np.stack([chunk_embeddings[cid] for cid in chunk_ids], axis=0).astype(np.float32)
        np.save(self.storage_path / "chunk_vecs.npy", chunk_vecs)
        
        # 2. 构建映射
        chunk_idx_map = {cid: i for i, cid in enumerate(chunk_ids)}
        chunk_vecs_mm = np.load(self.storage_path / "chunk_vecs.npy", mmap_mode="r")
        
        # 3. 为每个概念-关系对构建 leaf bucket
        concept_attr_chunks = defaultdict(lambda: defaultdict(list))
        for h, aid, t, cid in self.triples:
            if cid in chunk_idx_map:
                # 为源实体和目标实体都添加chunk
                concept_attr_chunks[h][aid].append(chunk_idx_map[cid])
                concept_attr_chunks[t][aid].append(chunk_idx_map[cid])
        
        # 4. 保存 leaf buckets
        leaf_dir = self.storage_path / "leaf_buckets"
        shutil.rmtree(leaf_dir, ignore_errors=True)
        leaf_dir.mkdir(parents=True)
        
        for cid, attr_chunks in concept_attr_chunks.items():
            # 为每个概念创建子目录
            cid_dir = leaf_dir / f"cid_{cid}"
            cid_dir.mkdir(exist_ok=True)
            
            # 保存关系到chunk的映射
            attr_map = {}
            for aid, idxs in attr_chunks.items():
                attr_map[str(aid)] = idxs
            
            with open(cid_dir / "attr_map.json", "w", encoding="utf-8") as f:
                json.dump(attr_map, f, ensure_ascii=False)
            
            # 为每个关系创建索引
            for aid, idxs in attr_chunks.items():
                uniq = sorted(set(idxs))
                if not uniq:
                    continue
                
                vecs = chunk_vecs_mm[uniq]
                d_dim = vecs.shape[1]
                
                index = faiss.IndexFlatIP(d_dim)
                index.add(vecs)
                
                # 保存该关系的索引
                faiss.write_index(index, str(cid_dir / f"aid_{aid}.flatip"))
        
        logging.info(f"Built {len(list(leaf_dir.glob('**/*.flatip')))} leaf buckets with concept-attribute pairs")
    
    def query_semantic(self, query: str, cid: Optional[int] = None, 
                      aid: Optional[int] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """语义相似度查询，支持按概念和关系过滤"""
        import faiss
        
        # 1. 计算查询向量
        query_vec = self.emb_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        query_vec = query_vec.astype(np.float32)
        
        # 2. 确定查询范围
        if cid is not None:
            # 如果指定了概念，只在该概念的leaf buckets中搜索
            cid_dir = self.storage_path / "leaf_buckets" / f"cid_{cid}"
            if not cid_dir.exists():
                return []
            
            if aid is not None:
                # 如果指定了关系，直接查询该关系的索引
                index_path = cid_dir / f"aid_{aid}.flatip"
                if not index_path.exists():
                    return []
                index = faiss.read_index(str(index_path))
                D, I = index.search(query_vec.reshape(1, -1), top_k)
                
                # 获取chunk IDs
                with open(cid_dir / "attr_map.json", "r") as f:
                    attr_map = json.load(f)
                chunk_ids = attr_map.get(str(aid), [])
                
                results = []
                for score, idx in zip(D[0], I[0]):
                    if idx < len(chunk_ids):
                        chunk_id = list(self.chunk_embeddings.keys())[chunk_ids[idx]]
                        results.append((chunk_id, float(score)))
                return results
            else:
                # 查询该概念下的所有关系
                results = []
                for index_path in cid_dir.glob("aid_*.flatip"):
                    index = faiss.read_index(str(index_path))
                    D, I = index.search(query_vec.reshape(1, -1), top_k)
                    
                    # 获取关系信息
                    aid = int(index_path.stem.split("_")[1])
                    with open(cid_dir / "attr_map.json", "r") as f:
                        attr_map = json.load(f)
                    chunk_ids = attr_map.get(str(aid), [])
                    
                    for score, idx in zip(D[0], I[0]):
                        if idx < len(chunk_ids):
                            chunk_id = list(self.chunk_embeddings.keys())[chunk_ids[idx]]
                            results.append((chunk_id, float(score)))
                
                # 按相似度排序并返回top_k
                return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        else:
            # 如果没有指定概念，在所有leaf buckets中搜索
            results = []
            for cid_dir in (self.storage_path / "leaf_buckets").glob("cid_*"):
                if not cid_dir.is_dir():
                    continue
                
                for index_path in cid_dir.glob("aid_*.flatip"):
                    index = faiss.read_index(str(index_path))
                    D, I = index.search(query_vec.reshape(1, -1), top_k)
                    
                    # 获取关系信息
                    aid = int(index_path.stem.split("_")[1])
                    with open(cid_dir / "attr_map.json", "r") as f:
                        attr_map = json.load(f)
                    chunk_ids = attr_map.get(str(aid), [])
                    
                    for score, idx in zip(D[0], I[0]):
                        if idx < len(chunk_ids):
                            chunk_id = list(self.chunk_embeddings.keys())[chunk_ids[idx]]
                            results.append((chunk_id, float(score)))
            
            # 按相似度排序并返回top_k
            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def visualize(self, output_file: str = "knowledge_graph.html"):
        """可视化知识图谱，展示 super community 和 cluster 结构"""
        from pyvis.network import Network
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import numpy as np
        
        # 1. 创建网络图
        net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # 2. 为每个 super community 创建子图
        super_communities = {}
        for cid, scid in self.cid2scid.items():
            if scid not in super_communities:
                super_communities[scid] = []
            super_communities[scid].append(cid)
        
        # 3. 为每个 super community 分配不同的颜色
        scid_colors = {}
        for scid in super_communities.keys():
            scid_colors[scid] = mcolors.to_hex(cm.tab20(scid % 20))
        
        # 4. 为每个 cluster 分配不同的形状
        cluster_shapes = {
            0: "dot",
            1: "triangle",
            2: "square",
            3: "diamond",
            4: "star",
            5: "hexagon"
        }
        
        # 5. 添加节点
        for cid, name in self.concept2cid.items():
            scid = self.cid2scid.get(cid, -1)
            cluster_id = self.cid2cluster.get(cid, -1)
            
            # 设置节点属性
            node_color = "#97c2fc" if scid == -1 else scid_colors[scid]
            node_shape = cluster_shapes.get(cluster_id % len(cluster_shapes), "dot")
            
            # 计算节点位置
            x = np.random.uniform(-100, 100) if scid == -1 else scid * 200 + np.random.uniform(-50, 50)
            y = np.random.uniform(-100, 100) if cluster_id == -1 else cluster_id * 200 + np.random.uniform(-50, 50)
            
            net.add_node(
                str(cid),
                label=name,
                title=f"Community: {scid}\nCluster: {cluster_id}",
                color=node_color,
                shape=node_shape,
                x=x,
                y=y,
                physics=True
            )
        
        # 6. 添加边
        for h, aid, t, _ in self.triples:
            if str(h) in net.get_nodes() and str(t) in net.get_nodes():
                # 根据源节点和目标节点的社区设置边的颜色
                h_scid = self.cid2scid.get(h, -1)
                t_scid = self.cid2scid.get(t, -1)
                edge_color = "#808080" if h_scid != t_scid else scid_colors.get(h_scid, "#808080")
                
                net.add_edge(
                    str(h),
                    str(t),
                    label=list(self.attr2aid.keys())[list(self.attr2aid.values()).index(aid)],
                    color=edge_color,
                    width=2 if h_scid == t_scid else 1
                )
        
        # 7. 配置布局和交互
        net.toggle_physics(True)
        net.show_buttons(['physics', 'nodes', 'edges'])
        
        # 8. 添加图例
        legend_html = """
        <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border: 1px solid #ccc;">
            <h3>Legend</h3>
            <div style="display: flex; flex-direction: column; gap: 5px;">
        """
        
        # 添加社区图例
        legend_html += "<h4>Super Communities</h4>"
        for scid, color in scid_colors.items():
            legend_html += f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background: {color}; margin-right: 5px;"></div>Community {scid}</div>'
        
        # 添加聚类图例
        legend_html += "<h4>Clusters</h4>"
        for shape_id, shape in cluster_shapes.items():
            legend_html += f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; margin-right: 5px;">{shape}</div>Cluster {shape_id}</div>'
        
        legend_html += "</div></div>"
        
        # 9. 保存可视化
        net.write_html(output_file)
        
        # 添加图例到HTML文件
        with open(output_file, 'r') as f:
            html_content = f.read()
        
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Knowledge graph visualization saved to {output_file}")
        return output_file

class KGProcessor:
    def __init__(self, storage_path: str, api_key: str, base_url: str):
        self.text_processor = TextProcessor()
        self.relation_extractor = RelationExtractor(api_key, base_url)
        self.storage = GraphStorage(storage_path)
        self.kg = self.storage.load_graph()
        
        self.kg_storage = KGStorage(storage_path)
        
        # 用于存储文档的嵌入向量
        self.chunk_embeddings = {}
    
    def _update_kg(self, extraction: Dict[str, Any]):
        """更新知识图谱，添加实体和关系"""
        # 1. 首先处理所有实体（包括关系中的实体）
        entities_to_add = set()
        
        # 1.1 添加显式声明的实体
        for entity_data in extraction["entities"]:
            entities_to_add.add((
                entity_data["entity_name"],
                entity_data["entity_type"],
                entity_data["entity_description"]
            ))
        
        # 1.2 添加关系中的实体
        for relation_data in extraction["relations"]:
            # 添加源实体
            entities_to_add.add((
                relation_data["source_entity"],
                "Entity",  # 默认类型
                f"Source entity in relation: {relation_data['relation']}"
            ))
            # 添加目标实体
            entities_to_add.add((
                relation_data["target_entity"],
                "Entity",  # 默认类型
                f"Target entity in relation: {relation_data['relation']}"
            ))
        
        # 1.3 添加所有实体到知识图谱
        for name, type, description in entities_to_add:
            if name not in self.kg.entity_name_to_id:
                entity = Entity(name=name, type=type, description=description)
                self.kg.entities[entity.id] = entity
                self.kg.entity_name_to_id[entity.name] = entity.id
        
        # 2. 处理所有关系
        for relation_data in extraction["relations"]:
            source = self.kg.entities[self.kg.entity_name_to_id[relation_data["source_entity"]]]
            target = self.kg.entities[self.kg.entity_name_to_id[relation_data["target_entity"]]]
            
            relation = Relation(
                source=source,
                relation=relation_data["relation"],
                target=target,
                description=relation_data["relation_description"]
            )
            self.kg.relations[relation.id] = relation
            self.kg.relation_triples.add((source.id, relation.relation, target.id))
    
    def process_document(self, doc_id: str, text: str):
        # 1. 清理和分块
        cleaned_text = self.text_processor.clean_text(text)
        chunks = self.text_processor.split_into_chunks(cleaned_text)
        
        # 2. 提取实体和关系
        for chunk, start, end in chunks:
            extraction = self.relation_extractor.extract_from_chunk(chunk)
            self._update_kg(extraction)
            
            # 计算并存储 chunk 的嵌入向量
            chunk_id = f"{doc_id}_{start}_{end}"
            self.chunk_embeddings[chunk_id] = self.kg_storage.emb_model.encode(
                chunk, normalize_embeddings=True, convert_to_numpy=True
            )
        
        # 3. 保存更新后的图谱
        self.storage.save_graph(self.kg)
    
    def finalize(self):
        """完成知识图谱的处理"""
        # 1. 构建映射
        self.kg_storage.build_mappings(self.kg)
        
        # 2. 计算嵌入向量
        self.kg_storage.compute_embeddings()
        
        # 3. 检测社区
        self.kg_storage.detect_communities()
        
        # 4. 聚类
        self.kg_storage.cluster_embeddings()
        
        # 5. 构建 leaf buckets
        self.kg_storage.build_leaf_buckets(self.chunk_embeddings)
        
        # 6. 生成可视化
        return self.kg_storage.visualize()
    
    def print_all_entities(kg):
        for eid, entity in kg.entities.items():
            print(f"{eid}: {entity.name}")
    
    def lookup_relation_chunk_embeddings(self, entity_name: str, relation_name: Optional[str] = None) -> List[Tuple[str, str, np.ndarray]]:
        results = []
        
        entity_name_lower = entity_name.lower().strip()
        cid = self.kg_storage.concept2cid.get(entity_name_lower)
        if cid is None:
            print(f"Concept for entity '{entity_name}' not found.")
            return results

        if relation_name:
            aid = self.kg_storage.attr2aid.get(relation_name.lower())
            if aid is None:
                print(f"Relation '{relation_name}' not found.")
                return results
        else:
            aid = None

        for h, a, t, chunk_id in self.kg_storage.triples:
            if (h == cid or t == cid) and (aid is None or a == aid):
                chunk_embedding = self.chunk_embeddings.get(chunk_id)
                if chunk_embedding is not None:
                    relation_str = list(self.kg_storage.attr2aid.keys())[list(self.kg_storage.attr2aid.values()).index(a)]
                    direction = "outgoing" if h == cid else "incoming"
                    results.append((direction, relation_str, chunk_embedding))
        
        if not results:
            print(f"No chunk embeddings found for entity '{entity_name}' with relation '{relation_name}'.")

        return results



def main():
    # 配置
    DATASET = "nfcorpus"
    STORAGE_PATH = "data"
    API_KEY = ""
    BASE_URL = "https://api.deepseek.com/v1"

    # 下载数据集
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, *_ = GenericDataLoader(data_folder=data_path).load(split="train")

    # 只取前3篇文档
    corpus = dict(list(corpus.items())[:2])
    logging.info(f"Processing first 3 documents from {DATASET} dataset")

    # 初始化处理器
    processor = KGProcessor(STORAGE_PATH, API_KEY, BASE_URL)
    processor.kg.print_entity_relations()
    # 处理文档
    for doc_id, record in corpus.items():
        logging.info(f"Processing document: {doc_id}")
        processor.process_document(doc_id, record["text"])
    
    KGProcessor.print_all_entities(processor.kg)
    
    # 生成知识图谱可视化
    output_file = processor.finalize()
    logging.info(f"Knowledge graph visualization generated: {output_file}")

    

if __name__ == "__main__":
    main() 