import re
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import SOURCE_DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, COLLECTION_NAME

def clean_text(text: str) -> str:
    """Text Cleaner: 基础清理与规范化"""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)  # 清除不可见特殊字符
    text = re.sub(r'\s+', ' ', text)  # 压缩连续空格
    return text.strip()

def parse_single_dtc_file(file_path: Path):
    documents = []
    brand_name = file_path.stem.split('_')[0].capitalize()
    pattern = re.compile(r"^([A-Z0-9]+)\s*-\s*(.+)$")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cleaned_line = clean_text(line)
            if not cleaned_line:
                continue
            
            match = pattern.match(cleaned_line)
            if match:
                dtc_code = match.group(1).upper()
                description = match.group(2)
                
                page_content = f"DTC Code: {dtc_code}\nBrand: {brand_name}\nDescription: {description}"
                metadata = {
                    "dtc_code": dtc_code,
                    "brand": brand_name,
                    "file_name": file_path.name
                }
                documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

def build_vector_store():
    if not SOURCE_DATA_DIR.exists():
        raise FileNotFoundError(f"未找到源数据目录: {SOURCE_DATA_DIR}")

    txt_files = list(SOURCE_DATA_DIR.glob("*.txt"))
    print(f"📂 找到 {len(txt_files)} 个源文件，开始解析与清洗...")

    all_documents = []
    for file_path in txt_files:
        docs = parse_single_dtc_file(file_path)
        all_documents.extend(docs)

    print(f"📊 累计提取 {len(all_documents)} 条清洗后的 Document 节点。加载 Embedding 模型中...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    print("✅ Chroma 向量库构建完成！")
    return vectorstore