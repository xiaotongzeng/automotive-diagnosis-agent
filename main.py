import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from data_pipeline.build_vector_store import build_all_indices
from src.agent.workflow import app
from config.settings import CHROMA_DB_DIR, DEEPSEEK_API_KEY


def main():
    # 1. 检查 LLM API Key
    if not DEEPSEEK_API_KEY:
        print("⚠️ 提醒：请通过 .env 或环境变量配置 LLM API Key！")
        return

    # 2. 检查向量数据库
    if not Path(CHROMA_DB_DIR).exists():
        print("⚡ 未找到 Chroma 向量数据库。")
        print("正在启动数据解析与向量索引构建...")
        build_all_indices()

    # 3. 运行 Agent 诊断
    user_query = "Acura 汽车报了 P1106 和 P1203 故障，是什么意思？"

    print(f"\n❓ 用户提问: {user_query}")

    inputs = {
        "messages": [
            HumanMessage(content=user_query)
        ],
        "vehicle_info": {}
    }

    result = app.invoke(inputs)

    # 4. 获取最终 Agent 回复
    messages = result.get("messages", [])

    print("\n🤖 Automotive Diagnosis Agent 诊断回复:\n")

    if messages:
        print(messages[-1].content)
    else:
        print("⚠️ Agent 没有返回消息。")


if __name__ == "__main__":
    main()
