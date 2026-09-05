import os
import re
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config.settings import (
    CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, COLLECTION_NAME, RERANKER_MODEL_NAME
)
from src.ingestion.hybrid_retriever import HybridRetriever

# 单例加载 Embeddings 与 Reranker
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
hybrid_retriever = HybridRetriever(vectorstore)
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)

def extract_clean_dtc(code_input: str) -> str:
    """从输入字符串中提取干净的 DTC 代码，例如从 '2020 Camry P0301 故障' 中提取 'P0301'"""
    match = re.search(r'[PBCU]\d{4}', code_input.upper())
    return match.group(0) if match else code_input.strip()

def _execute_deep_schema_retrieval(query: str, top_k: int = 3) -> Dict[str, Any]:
    """执行针对深层 Schema 数据的检索与 Rerank"""
    candidates = hybrid_retriever.search(query, top_k=10)
    
    if not candidates:
        return {"text": "未检索到匹配的深层诊断 Schema 知识。", "evidences": []}
    
    # Cross-Encoder 精排
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker_model.predict(pairs)
    scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    evidences = []
    text_outputs = []
    for idx, (doc, score) in enumerate(scored_docs, 1):
        source_dtc = doc.metadata.get("dtc", "DTC")
        evidences.append({
            "citation_id": f"Source-{idx}",
            "dtc": source_dtc,
            "rerank_score": float(score),
            "content": doc.page_content
        })
        text_outputs.append(f"【诊断依据 Source-{idx} | DTC: {source_dtc} | 相关度: {score:.2f}】\n{doc.page_content}")
        
    return {
        "text": "\n\n".join(text_outputs),
        "evidences": evidences
    }

@tool
def lookup_dtc_schema(code: str) -> str:
    """查询特定 DTC 故障码的全链路诊断 Schema，包含故障现象、根因分析、带标准公差(Expected Range)的排查步骤、维修方案及后处理验证。"""
    clean_code = extract_clean_dtc(code)
    res = _execute_deep_schema_retrieval(f"DTC 故障码 {clean_code} 深层诊断 Schema 排查步骤 测量标准", top_k=2)
    return res["text"]

@tool
def search_repair_procedure(code: str) -> str:
    """查询特定故障码的标准排查流程与物理测量参数公差。"""
    clean_code = extract_clean_dtc(code)
    res = _execute_deep_schema_retrieval(f"故障码 {clean_code} 测量步骤 Expected Range 标准范围", top_k=2)
    return res["text"]

@tool
def search_parts(code: str) -> str:
    """查询特定故障码涉及的替换备件及 OEM 件号。"""
    clean_code = extract_clean_dtc(code)
    res = _execute_deep_schema_retrieval(f"故障码 {clean_code} 替换备件 OEM 件号", top_k=2)
    return res["text"]

# 导出供 Agent 调用的工具列表
agent_tools = [lookup_dtc_schema, search_repair_procedure, search_parts]