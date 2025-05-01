import os
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import random
import logging
from kg_forest_new import TextProcessor, KnowledgeGraphBuilder, Entity, Relation, ChunkExtraction
from typing import Dict, Set, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_key = ""
kg_builder = KnowledgeGraphBuilder.load_graph(api_key=API_key, path_prefix="saved/graph")

def load_dataset(dataset_name: str = "nfcorpus") -> tuple:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, *_ = GenericDataLoader(data_folder=data_path).load(split="train")
    return corpus, data_path

def process_document(doc: tuple, text_processor: TextProcessor) -> dict:
    return {
        "id": doc[0],
        "title": text_processor.clean(doc[1]["title"]),
        "text": text_processor.clean(doc[1]["text"])
    }

def create_tasks(clean_doc: dict, text_processor: TextProcessor, kg: KnowledgeGraphBuilder) -> list:
    """
    根据文档创建任务列表
    
    Args:
        clean_doc: 清理后的文档
        text_processor: TextProcessor实例
        kg: KnowledgeGraphBuilder实例
        
    Returns:
        list: 任务列表
    """
    chunks = text_processor.split_into_chunks(clean_doc["text"])
    tasks = []
    
    for i, chunk in enumerate(chunks):
        chunk_index = f"{clean_doc['id']}#{i}"
        chunk_embedding = kg.embeddings[chunk_index]
        chunk_kg = kg.extractions[chunk_index]
        
        tasks.append({
            "chunk_id": chunk_index,
            "doc_id": clean_doc["id"],
            "title": clean_doc["title"],
            "start": chunk[1],
            "end": chunk[2],
            "origin_chunk": chunk[0],
            "embedding_chunk": chunk_embedding,
            "chunk-kg": chunk_kg,
        })
    
    return tasks

def extract_entities_and_relations(kg_data: Dict[str, ChunkExtraction]) -> Tuple[Set[Tuple[str, str, str]], Set[Tuple[str, str, str, str]]]:
    # Extract all unique entities
    all_entities = set()
    for chunk in kg_data.values():
        for entity in chunk.entities:
            all_entities.add((entity.entity_name, entity.entity_type, entity.entity_description))
    
    # Extract all relations
    all_relations = set()
    for chunk in kg_data.values():
        for relation in chunk.relations:
            all_relations.add((relation.source_entity, relation.relation, relation.target_entity, relation.relation_description))
    
    return all_entities, all_relations

def print_entities_and_relations(entities: Set[Tuple[str, str, str]], relations: Set[Tuple[str, str, str, str]]) -> None:
    # Print entities
    print("\n=== Entities ===")
    for entity in sorted(entities):
        print(f"Name: {entity[0]}")
        print(f"Type: {entity[1]}")
        print(f"Description: {entity[2]}")
        print("-" * 50)
    
    # Print relations
    print("\n=== Relations ===")
    for relation in sorted(relations):
        print(f"Source: {relation[0]}")
        print(f"Relation: {relation[1]}")
        print(f"Target: {relation[2]}")
        print(f"Description: {relation[3]}")
        print("-" * 50)

def update(tasks):
    """
    将从"云端"同步下来的 tasks（每个对应一个 chunk）：
      - 注册到 kg_builder 的内存结构
      - 把实体/关系插入到多级索引里
      - 最后重建 FAISS 索引
      
    """
    pre_entities = set(kg_builder.index.entity_name_to_id.keys())
    pre_relations = set(kg_builder.index.relation_name_to_id.keys())
    pre_chunks   = set(kg_builder.chunk_meta.keys())
    # 1) 把每个 chunk 的元数据、抽取结果、预计算 embedding 写入 kg_builder
    for task in tasks:
        chunk_id = task["chunk_id"]
        clean_doc_id = task["doc_id"]
        start, end = task["start"], task["end"]
        chunk_kg: ChunkExtraction = task["chunk-kg"]
        chunk_vec: np.ndarray = task["embedding_chunk"]
        
        # 更新元数据
        kg_builder.chunk_meta[chunk_id] = {
            "doc": clean_doc_id,
            "start": start,
            "end": end
        }
        # 保存抽取结果
        kg_builder.extractions[chunk_id] = chunk_kg
        # 保存 chunk 文本的 embedding
        kg_builder.embeddings[chunk_id] = chunk_vec
        
        # 2) 将这段 chunk 的实体/关系加入多级索引
        kg_builder.index.process_chunk_extraction(
            chunk_id,
            chunk_kg,
            entity_embedder=kg_builder.entity_embedder,
            relation_embedder=kg_builder.relation_embedder,
            incremental=True,      # 使用增量方式添加向量
            auto_flush=False       # 批量结束后再 flush
        )
    
    # 3) 一次性把新增向量写进 FAISS
    if getattr(kg_builder.index, "_new_vecs", None):
        kg_builder.index._flush_new_vectors()   # 内部会 build 或 add
    
    logging.info(f"Updated local KG with {len(tasks)} new chunks.")
    # （可选）持久化到磁盘
    # kg_builder.save_graph("saved/graph")
    # 更新后状态
    post_entities = set(kg_builder.index.entity_name_to_id.keys())
    post_relations = set(kg_builder.index.relation_name_to_id.keys())
    post_chunks   = set(kg_builder.chunk_meta.keys())

    new_entities  = post_entities  - pre_entities
    new_relations = post_relations - pre_relations
    new_chunks    = post_chunks    - pre_chunks

    logging.info(f"新增实体: {new_entities}")
    logging.info(f"新增关系: {new_relations}")
    logging.info(f"新增 chunk_id: {new_chunks}")


def main():
    try:
        # 加载数据集
        logger.info("Loading dataset...")
        corpus, _ = load_dataset()
        
        # 初始化处理器
        text_processor = TextProcessor()
        kg = KnowledgeGraphBuilder(api_key=API_key)
        
        # 随机选择文档
        doc = random.choice(list(corpus.items()))
        logger.info(f"Selected document ID: {doc[0]}")
        
        # 处理文档
        clean_doc = process_document(doc, text_processor)
        logger.info("Document processed successfully")
        
        # 处理文本并构建知识图谱
        kg.process_text(clean_doc["text"], clean_doc["id"])
        
        # 创建任务
        tasks = create_tasks(clean_doc, text_processor, kg)
        logger.info(f"Created {len(tasks)} tasks")
        
        update(tasks)

        kg_builder.visualize(output_file="updated_graph.html")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()