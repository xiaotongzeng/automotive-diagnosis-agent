import pickle
import jieba
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config.settings import BM25_INDEX_PATH

class HybridRetriever:
    def __init__(self, vectorstore, documents: list[Document] = None):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = None
        
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.documents = data["documents"]
        elif documents:
            self.build_bm25(documents)

    def build_bm25(self, documents: list[Document]):
        """构建并持久化 BM25 索引"""
        corpus = [list(jieba.cut(doc.page_content)) for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        self.documents = documents
        
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        print("✅ BM25 关键字索引构建并保存成功！")

    def search(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[Document]:
        """Hybrid Search: Dense + BM25，使用 Reciprocal Rank Fusion (RRF) 融合。"""
        # 1. 向量检索
        dense_docs = self.vectorstore.similarity_search(query, k=top_k, filter=filters)
        
        # 2. BM25 检索
        if not self.bm25:
            return dense_docs
        
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        sparse_docs = []
        for i in ranked_indices:
            doc = self.documents[i]
            if filters and any(doc.metadata.get(key) != value for key, value in filters.items()):
                continue
            sparse_docs.append(doc)
            if len(sparse_docs) == top_k:
                break
        
        # 3. RRF 融合，避免某一检索器的分数尺度主导最终排序。
        rrf_k = 60
        scores = {}
        docs_by_key = {}
        for ranked_docs in (dense_docs, sparse_docs):
            for rank, doc in enumerate(ranked_docs, start=1):
                key = doc.page_content
                docs_by_key[key] = doc
                scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
        return [docs_by_key[key] for key in sorted(scores, key=scores.get, reverse=True)[:top_k]]
