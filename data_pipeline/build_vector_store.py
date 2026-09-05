import json
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import BASE_DIR, EMBEDDING_MODEL_NAME, CHROMA_DB_DIR, COLLECTION_NAME
from src.ingestion.hybrid_retriever import HybridRetriever

PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_jsonl_documents(file_path):
    docs = []
    if not file_path.exists():
        return docs
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            docs.append(Document(page_content=data["content"], metadata=data["metadata"]))
    return docs

def build_all_indices(
    reset: bool = False,
    batch_size: int = 32,
    start: int = 0,
    end: int | None = None,
    build_bm25: bool = True,
):
    print("📂 正在加载处理后的 JSONL 数据...")
    dtc_docs = load_jsonl_documents(PROCESSED_DIR / "dtc_documents.jsonl")
    repair_docs = load_jsonl_documents(PROCESSED_DIR / "repair_documents.jsonl")
    parts_docs = load_jsonl_documents(PROCESSED_DIR / "parts_documents.jsonl")

    all_documents = dtc_docs + repair_docs + parts_docs
    print(f"📊 汇总文档数: {len(all_documents)} 条 (DTC: {len(dtc_docs)}, 维修: {len(repair_docs)}, 配件: {len(parts_docs)})")

    print("⚡ 加载 BGE-M3 Embedding 模型构建 Chroma 向量库...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if not all_documents:
        raise ValueError("没有可索引的文档；请先检查数据处理结果。")

    if not 0 <= start <= len(all_documents):
        raise ValueError(f"start 必须介于 0 和 {len(all_documents)} 之间。")
    end = len(all_documents) if end is None else min(end, len(all_documents))
    if end < start:
        raise ValueError("end 不能小于 start。")
    if reset and start:
        raise ValueError("reset 只能从第 0 条文档开始执行。")

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    if reset:
        vectorstore.delete_collection()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    indexed_documents = all_documents[start:end]
    # 同一 DTC 可能在多个来源中拥有完全相同正文；将稳定的全局序号纳入 ID，
    # 既避免 Chroma ID 冲突，也使同一数据版本可重复构建。
    ids = [
        hashlib.sha256(f"{start + offset}\0{doc.page_content}".encode("utf-8")).hexdigest()
        for offset, doc in enumerate(indexed_documents)
    ]
    total_batches = (len(indexed_documents) + batch_size - 1) // batch_size
    for batch_number, offset in enumerate(range(0, len(indexed_documents), batch_size), start=1):
        batch_end = offset + batch_size
        vectorstore.add_documents(indexed_documents[offset:batch_end], ids=ids[offset:batch_end])
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            current = start + min(batch_end, len(indexed_documents))
            print(f"   已写入批次 {batch_number}/{total_batches}（全量 {current}/{len(all_documents)}）", flush=True)
    print("✅ Chroma DB 构建成功！")

    if build_bm25:
        print("⚡ 正在构建全局 BM25 索引...")
        retriever = HybridRetriever(vectorstore)
        retriever.build_bm25(all_documents)
        print("🎉 全局数据库与 BM25 索引构建完毕！")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="删除现有 collection 后重建索引")
    parser.add_argument("--batch-size", type=int, default=32, help="每批嵌入文档数")
    parser.add_argument("--start", type=int, default=0, help="续建时的起始文档偏移")
    parser.add_argument("--end", type=int, help="续建时的结束文档偏移（不含）")
    parser.add_argument("--skip-bm25", action="store_true", help="仅写入向量，暂不重建 BM25")
    args = parser.parse_args()
    build_all_indices(
        reset=args.reset,
        batch_size=args.batch_size,
        start=args.start,
        end=args.end,
        build_bm25=not args.skip_bm25,
    )
