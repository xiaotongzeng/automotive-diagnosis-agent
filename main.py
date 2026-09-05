import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from data_pipeline.build_vector_store import build_all_indices
from src.agent.workflow import app
from config.settings import CHROMA_DB_DIR, QWEN_API_KEY

def main():
    # 1. 检查环境变量 API Key
    if not QWEN_API_KEY:
        print("⚠️ 提醒：请通过 .env 或环境变量配置 DASHSCOPE_API_KEY！")
    
    # 2. 数据库检查与自动初始化
    if not Path(CHROMA_DB_DIR).exists():
        print("⚡ 数据库未找到，启动多文件批量数据解析与向量写入...")
        build_all_indices()
    
    # 3. 运行在线 Agent 诊断
    user_query = " Acura 汽车报了 P1106 和 P1203 故障，是什么意思？"
    print(f"\n❓ 用户提问: {user_query}")
    
    result = app.invoke({"question": user_query})
    print("\n🤖 通义千问 AI 诊断回复:\n")
    print(result["generation"])

if __name__ == "__main__":
    main()
