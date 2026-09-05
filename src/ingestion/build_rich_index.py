import os
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, COLLECTION_NAME
from src.ingestion.schema_converter import StandardDTCRecord

def build_structured_knowledge_base(json_filepath: str):
    print("📦 开始加载新 Schema 结构化数据...")
    with open(json_filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    documents = []
    for item in raw_data:
        record = StandardDTCRecord(**item)
        page_content = record.to_rag_document_text()
        
        # 将结构化字段抽离为 metadata，便于工具精准过滤或直接获取字段
        metadata = {
            "dtc": record.dtc,
            "system": record.system,
            "component": record.component,
            "doc_type": "deep_diagnostic_schema",
            "has_steps": len(record.diagnostic_steps) > 0
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
        
    print(f"✅ 成功构建 {len(documents)} 条深层诊断 Document 对象。")
    
    # 建立向量数据库索引
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    print("🚀 结构化深层 RAG 数据库创建并持久化完成！")

if __name__ == "__main__":
    # 示例入口
    build_structured_knowledge_base("data/structured_dtc_database.json")