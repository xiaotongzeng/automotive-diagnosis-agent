import os
from pathlib import Path
from dotenv import load_dotenv

# 0. 底层死锁与并发环境变量防护（解决 macOS / Python 3.13 下的 Segfault 与卡死）
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. 强制加载根目录下的 .env 文件（配置 override=True 防止缓存或旧环境变量覆盖）
load_dotenv(BASE_DIR / ".env", override=True)

# 2. 路径配置
SOURCE_DATA_DIR = BASE_DIR / "data" / "raw" / "dtc_codes"
STRUCTURED_DATABASE_PATH = BASE_DIR / "data" / "structured_dtc_database.json"
CHROMA_DB_DIR = str(BASE_DIR / "vector_store" / "chroma_db")
BM25_INDEX_PATH = BASE_DIR / "vector_store" / "bm25_index.pkl"

# 3. Embedding & Reranker 检索模型配置
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
COLLECTION_NAME = os.getenv("CAR_RAG_COLLECTION", "car_dtc_collection_v2")

# 4. LLM 与 DeepSeek API 配置
# 兼顾 DEEPSEEK_API_KEY、OPENAI_API_KEY 与 DASHSCOPE_API_KEY
DEEPSEEK_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY") 
    or os.getenv("OPENAI_API_KEY") 
    or os.getenv("DASHSCOPE_API_KEY", "")
).strip()

# 如果系统环境缺少 OPENAI_API_KEY，自动将读取到的 Key 注入到系统环境变量中
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY

DEEPSEEK_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")  # 可切换为 deepseek-reasoner

# 兼容项目原有变量名
QWEN_BASE_URL = DEEPSEEK_BASE_URL