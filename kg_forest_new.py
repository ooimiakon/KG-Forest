from __future__ import annotations
import os
import json
import re
import logging
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import openai
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
from kg_index import MultiLevelIndex
import faiss
import pickle

# Global variables
API_KEY = ""
MAX_PROCESSED_SAMPLES = 3  # Maximum number of samples to process

# Set environment variables to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

@dataclass
class Entity:
    """实体数据结构"""
    entity_name: str        # 实体名称
    entity_type: str        # 实体类型（如Person, Organization, Location等）
    entity_description: str # 实体描述

@dataclass
class Relation:
    """关系数据结构"""
    source_entity: str   # 关系起点实体
    relation: str        # 关系类型
    target_entity: str   # 关系终点实体
    relation_description: str     # 关系描述

@dataclass
class ChunkExtraction:
    """文本块抽取结果"""
    entities: List[Entity]    # 从文本块中抽取的所有实体
    relations: List[Relation] # 从文本块中抽取的所有关系

class TextProcessor:
    """文本处理器：负责文本清理和分块"""
    
    def __init__(self, max_chars: int = 600):
        """
        Args:
            max_chars: 每个文本块的最大字符数
        """
        self.max_chars = max_chars
        self.punct_re = re.compile(r'[\.!?]')  # 句子结束标点符号的正则表达式
    
    def clean(self, text: str) -> str:
        """
        清理文本：移除HTML标签和多余空格
        
        Args:
            text: 输入文本
        Returns:
            清理后的文本
        """
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def split_into_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """
        将文本分割成块，确保在句子边界处分割
        
        Args:
            text: 输入文本
        Returns:
            List[Tuple[chunk_text, start_index, end_index]]
        """
        text = text.strip()
        spans = []
        i, L = 0, len(text)
        while i < L:
            end = min(L, i + self.max_chars)
            m = self.punct_re.search(text, end)
            if m:
                end = m.end()
            chunk = text[i:end].strip()
            spans.append((chunk, i, end))
            i = end
        logging.info(f"Created {len(spans)} chunks from text (max_chars={self.max_chars})")
        return spans

class DeepSeekExtractor:
    """DeepSeek API封装：负责实体关系抽取"""
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: DeepSeek API密钥
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.tools = self._setup_tools()
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
    
    def _setup_tools(self) -> List[Dict]:
        """设置DeepSeek API的工具定义"""
        return [{
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
        }]

    def extract_chunk(self, chunk: str) -> ChunkExtraction:
        """
        从文本块中抽取实体和关系
        
        Args:
            chunk: 输入的文本块
        Returns:
            ChunkExtraction对象，包含抽取的实体和关系
        """
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.few_shot + [{"role": "user", "content": f'Chunk:\n"{chunk}"'}],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "extract_entities_relations"}},
                temperature=0.0
            )
            if resp.choices[0].message.tool_calls:
                result = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
                return ChunkExtraction(
                    entities=[Entity(**e) for e in result["entities"]],
                    relations=[Relation(**{k: v for k, v in r.items() if k in {'source_entity', 'relation', 'target_entity', 'relation_description'}}) for r in result["relations"]]
                )
            raise Exception("No tool calls in response")
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            raise

class KnowledgeGraphBuilder:
    """主要是得到 chunk_id -> 抽取结果"""
    
    def __init__(self, api_key: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            api_key: DeepSeek API密钥
            model_name: 用于文本嵌入的模型名称
        """
        self.extractor = DeepSeekExtractor(api_key)
        self.processor = TextProcessor()
        self.index = MultiLevelIndex()  # 使用默认配置
        self.entity_embedder = SentenceTransformer(model_name)
        self.relation_embedder = SentenceTransformer(model_name)
        self.chunk_meta: Dict[str, Dict] = {}      # chunk_id -> {doc, start, end}
        self.extractions: Dict[str, ChunkExtraction] = {}  # chunk_id -> 抽取结果
        self.embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> 文本嵌入向量
    
    def process_text(self, text: str, doc_id: str, title: str = None) -> List[ChunkExtraction]:
        """
        Process text into chunks and extract entities and relations.
        
        Args:
            text: 输入文本
            doc_id: 文档ID
            title: 文档标题（可选）
        Returns:
            List[ChunkExtraction]: 抽取结果列表
        """
        # 清理文本
        text = self.processor.clean(text)
        logging.info(f"[DEBUG] process_text called with doc_id={doc_id}, title={title}")
        
        # 分割成块
        chunks = self.processor.split_into_chunks(text)
        extractions = []
        
        for i, (chunk, start, end) in enumerate(chunks):
            chunk_id = f"{doc_id}#{i}"  # 使用文档ID和块索引生成唯一的chunk_id
            extraction = self.extractor.extract_chunk(chunk)
            extractions.append(extraction)
            
            # 保存元数据和抽取结果
            logging.info(f"[DEBUG] Creating metadata for chunk {chunk_id} with title={title}")
            meta = {
                "doc": doc_id,
                "title": title,  # Add title to metadata
                "start": start,
                "end": end
            }
            logging.info(f"[DEBUG] Metadata before storage: {meta}")
            self.chunk_meta[chunk_id] = meta
            logging.info(f"[DEBUG] Metadata after storage: {self.chunk_meta[chunk_id]}")
            self.extractions[chunk_id] = extraction
            
            # 生成嵌入向量
            embedding = self.entity_embedder.encode(
                chunk,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)
            self.embeddings[chunk_id] = embedding
            
        return extractions
    
    def process_document(self, doc_id: str, text: str, title: str = None, extractor=None):
        """
        处理单个文档
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            title: 文档标题（可选）
            extractor: DeepSeek抽取器实例（可选）
        """
        if extractor is None:
            extractor = self.extractor
            
        # 处理文本
        logging.info(f"[DEBUG] process_document called with doc_id={doc_id}, title={title}")
        self.process_text(text, doc_id, title)
        
        logging.info(f"Processed document {doc_id}")
    
    def build_index(self):
        """构建索引 - 在所有文档处理完成后调用"""
        
        for chunk_id, extraction in self.extractions.items():
            # 添加到索引
            self.index.process_chunk_extraction(
                chunk_id,
                extraction,
                self.entity_embedder,
                self.relation_embedder
            )
        logging.info(f"Building index with {len(self.extractions)} data points...")
        self.index.build_faiss_indices()
        logging.info("Index built successfully.")
    
    def query_graph(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Query the knowledge graph using semantic search."""
        # 获取查询嵌入
        query_embedding = self.entity_embedder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # 在索引中搜索
        entity_ids, distances = self.index.entity_index.search(
            query_embedding.reshape(1, -1),
            top_k
        )
        
        # 转换为实体名称和分数
        results = []
        for entity_id, distance in zip(entity_ids[0], distances[0]):
            entity_id = int(entity_id)  # 确保是整数
            entity_name = self.index.entity_id_to_name[entity_id]
            score = 1.0 / (1.0 + distance)  # 将距离转换为相似度分数
            results.append((entity_name, score))
            
        return results
    
    def visualize(self, output_file: str = "knowledge_graph.html") -> None:
        """
        将知识图谱可视化为HTML文件
        
        Args:
            output_file: 输出的HTML文件路径
        """
        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # 收集所有唯一节点
        all_nodes = set()
        for extraction in self.extractions.values():
            for entity in extraction.entities:
                all_nodes.add(entity.entity_name)
            for relation in extraction.relations:
                all_nodes.add(relation.source_entity)
                all_nodes.add(relation.target_entity)
        
        # 添加节点和边
        for node in all_nodes:
            net.add_node(node)
            
        for extraction in self.extractions.values():
            for relation in extraction.relations:
                net.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    title=relation.relation_description,
                    label=relation.relation
                )
        
        net.toggle_physics(True)
        net.show_buttons(['physics'])
        net.write_html(output_file)
        print(f"Knowledge graph saved to {output_file}")
    
    def save_graph(self, path_prefix: str):
        """Save metadata and FAISS index to disk"""
        logging.info(f"[DEBUG] Saving chunk_meta with titles: {[(k, v.get('title')) for k, v in self.chunk_meta.items()]}")
        metadata = {
            "chunk_meta": self.chunk_meta,
            "extractions": self.extractions,
            "embeddings": self.embeddings,
            "entity_name_to_id": self.index.entity_name_to_id,
            "entity_id_to_name": self.index.entity_id_to_name,
            "relation_name_to_id": self.index.relation_name_to_id,
            "entity_vectors": self.index.entity_vectors,
            "relation_vectors": self.index.relation_vectors,
            "entity_to_relations": self.index.entity_to_relations,
            "entity_relation_to_chunks": self.index.entity_relation_to_chunks,
        }
        import os
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

        with open(f"{path_prefix}.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"[✓] Saved metadata to {path_prefix}.pkl")

        if self.index.entity_index is not None:
            faiss.write_index(self.index.entity_index, f"{path_prefix}.faiss")
            print(f"[✓] Saved FAISS index to {path_prefix}.faiss")
        else:
            print("Warning: No FAISS index to save.")
    
    # load graph
    @staticmethod
    def load_graph(api_key: str, path_prefix: str) -> KnowledgeGraphBuilder:
        """
        Load a previously saved KnowledgeGraphBuilder from disk.
        
        Args:
            api_key: DeepSeek API key.
            path_prefix: Path prefix used when saving the graph (e.g., "saved/graph").
        
        Returns:
            A fully restored KnowledgeGraphBuilder instance.
        """
        import pickle
        import faiss

        # Step 1: Create an empty instance
        kg_builder = KnowledgeGraphBuilder(api_key=api_key)

        # Step 2: Load metadata
        with open(f"{path_prefix}.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Step 3: Restore internal state
        kg_builder.chunk_meta = metadata["chunk_meta"]
        logging.info(f"[DEBUG] Loaded chunk_meta with titles: {[(k, v.get('title')) for k, v in kg_builder.chunk_meta.items()]}")
        kg_builder.extractions = metadata["extractions"]
        kg_builder.embeddings = metadata["embeddings"]
        kg_builder.index.entity_name_to_id = metadata["entity_name_to_id"]
        kg_builder.index.entity_id_to_name = metadata["entity_id_to_name"]
        kg_builder.index.relation_name_to_id = metadata["relation_name_to_id"]
        kg_builder.index.entity_vectors = metadata["entity_vectors"]
        kg_builder.index.relation_vectors = metadata["relation_vectors"]
        kg_builder.index.entity_to_relations = metadata["entity_to_relations"]
        kg_builder.index.entity_relation_to_chunks = metadata["entity_relation_to_chunks"]

        # Step 4: Load FAISS index
        kg_builder.index.entity_index = faiss.read_index(f"{path_prefix}.faiss")

        print(f"[✓] Knowledge graph loaded from {path_prefix}.pkl and {path_prefix}.faiss")
        return kg_builder

    def validate_entity_extraction(self, query: str) -> None:
        """
        Validate entity extraction from a query.
        
        Args:
            query: The input query to validate
        """
        print("\n=== Entity Extraction Validation ===")
        print(f"Query: {query}")
        
        # Extract entities and relations
        entities, relations = self.extractor.extract_chunk(query)
        
        print("\nExtracted Entities:")
        for entity in entities:
            print(f"- Name: {entity.entity_name}")
            print(f"  Type: {entity.entity_type}")
            print(f"  Description: {entity.entity_description}")
            print()
            
        print("\nExtracted Relations:")
        for relation in relations:
            print(f"- Source: {relation.source_entity}")
            print(f"  Relation: {relation.relation}")
            print(f"  Target: {relation.target_entity}")
            print(f"  Description: {relation.relation_description}")
            print()

    def validate_embeddings(self, query: str) -> None:
        """
        Validate embedding generation for a query.
        
        Args:
            query: The input query to validate
        """
        print("\n=== Embedding Validation ===")
        print(f"Query: {query}")
        
        # Generate embeddings
        query_embedding = self.entity_embedder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        print(f"\nEmbedding shape: {query_embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(query_embedding)}")
        
        # Test similarity with some known entities
        if self.index.entity_vectors:
            similarities = []
            for entity_id, entity_name in self.index.entity_id_to_name.items():
                if entity_id < len(self.index.entity_vectors):
                    entity_vec = self.index.entity_vectors[entity_id]
                    similarity = np.dot(query_embedding, entity_vec) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entity_vec)
                    )
                    similarities.append((entity_name, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop 5 most similar entities:")
            for entity_name, similarity in similarities[:5]:
                print(f"- {entity_name}: {similarity:.3f}")

    def validate_index(self) -> None:
        """
        Validate the knowledge graph index.
        """
        print("\n=== Index Validation ===")
        
        print("\nEntity Statistics:")
        print(f"Total entities: {len(self.index.entity_id_to_name)}")
        print(f"Total relations: {len(self.index.relation_name_to_id)}")
        
        print("\nEntity-Relation Mapping:")
        for entity_id, relations in self.index.entity_to_relations.items():
            entity_name = self.index.entity_id_to_name.get(entity_id, "Unknown")
            print(f"\nEntity: {entity_name}")
            for rel_id in relations:
                rel_name = self.index.relation_name_to_id.get(rel_id, "Unknown")
                chunks = self.index.entity_relation_to_chunks.get((entity_id, rel_id), set())
                print(f"- Relation: {rel_name}")
                print(f"  Connected chunks: {len(chunks)}")
        
        if self.index.entity_index:
            print("\nFAISS Index Statistics:")
            print(f"Total vectors: {self.index.entity_index.ntotal}")
            print(f"Dimension: {self.index.entity_index.d}")

def main():
    """Main function: Demonstrates how to use the knowledge graph builder"""
    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S',
                       level=logging.INFO)
    
    # Initialize components
    kg_builder = KnowledgeGraphBuilder(api_key=API_KEY)
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载train.json
    train_path = os.path.join(current_dir, "datasets", "MultiHopRAG", "train.json")
    print(f"Loading train data from: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 加载corpus.json
    corpus_path = os.path.join(current_dir, "datasets", "MultiHopRAG", "corpus.json")
    print(f"Loading corpus data from: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    # 创建标题到文档的映射
    corpus_by_title = {doc['title']: doc for doc in corpus_data}
    
    # 处理前MAX_PROCESSED_SAMPLES条evidence_list
    print(f"\nProcessing first {MAX_PROCESSED_SAMPLES} evidence lists...")
    processed_count = 0
    seen_titles = set()
    for sample in train_data[:MAX_PROCESSED_SAMPLES]:           # 只取前 3 个样本
            
        for evidence in sample['evidence_list']:
            title = evidence['title']
            if title in seen_titles:        # 避免同一篇重复处理
                continue
            seen_titles.add(title)
            if title not in corpus_by_title:
                continue
            logging.info(f"[DEBUG] Processing evidence with title: {title}")
            doc = corpus_by_title[title]
            doc_id = f"doc_{processed_count}"
                
            # 将文章信息组合成文本
            text = f"Title: {doc['title']}\nAuthor: {doc['author']}\nSource: {doc['source']}\nCategory: {doc['category']}\nPublished: {doc['published_at']}\nFact: {doc['body']}"
            print(f"\nProcessing document {processed_count + 1}:")
            print(f"Title: {doc['title']}")
            logging.info(f"[DEBUG] Calling process_document with doc_id={doc_id}, title={doc['title']}")
            kg_builder.process_document(doc_id, text, title=doc['title'])
            processed_count += 1
    
    # 构建索引
    print("\nBuilding knowledge graph index...")
    kg_builder.build_index()
    
    # 可视化知识图谱
    output_file = os.path.join(current_dir, "knowledge_graph.html")
    print(f"\nGenerating visualization...")
    kg_builder.visualize(output_file=output_file)
    
    # Save the built knowledge graph
    save_dir = os.path.join(current_dir, "saved")
    os.makedirs(save_dir, exist_ok=True)
    print("\nSaving knowledge graph...")
    kg_builder.save_graph(os.path.join(save_dir, "graph"))
    print("Done!")

if __name__ == "__main__":
    main()