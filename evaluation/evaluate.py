import os
import sys
import logging
from pathlib import Path

# 0. 顶层环境变量隔离（禁用多进程抢占锁）
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
except Exception:
    pass

# 1. 项目路径与环境变量加载
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env", override=True)

# 确保环境变量注入 API Key
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["DEEPSEEK_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key

# 2. 预热 jieba
import jieba
jieba.initialize()

# 3. 导入 Agent 工作流与消息类
from typing import List, Dict, Any, Union
from src.agent.workflow import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage

# 调整测试用例，支持中文知识库关键词同义词
TEST_DATASET = [
    {
        "category": "DTC_Exact",
        "question": "P0301 故障码是什么含义？",
        "expected_tool": "lookup_dtc_schema",
        "expected_keywords": ["cylinder 1", "气缸 1", "1 缸", "第 1 缸", "失火", "缺火"]
    },
    {
        "category": "Vehicle_Context_Repair",
        "question": "2020 Toyota Camry 2.5L 报 P0301，怠速抖动严重，排查步骤是什么？",
        "expected_tool": "search_repair_procedure",
        "expected_keywords": ["点火", "火花塞", "点火线圈", "测量"]
    },
    {
        "category": "Parts_Lookup",
        "question": "解决 P0301 故障通常需要准备哪些零件？",
        "expected_tool": "search_parts",
        "expected_keywords": ["火花塞", "点火线圈", "喷油嘴"]
    }
]

def extract_content_string(content: Any) -> str:
    """提取各种数据类型下的文本内容"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)

def run_comprehensive_eval():
    print("🧪 开始运行多维度 Evaluation 评估套件...\n")
    total = len(TEST_DATASET)
    tool_correct = 0
    keyword_hit = 0

    for idx, item in enumerate(TEST_DATASET, 1):
        print(f"[{idx}/{total}] 测试项类别: {item['category']}")
        print(f"   👉 Question: {item['question']}")
        
        inputs = {
            "messages": [HumanMessage(content=item['question'])],
            "vehicle_info": {"brand": "Toyota", "model": "Camry", "year": "2020", "engine": "2.5L"}
        }
        
        try:
            print("   ⏳ 正在调用 Agent 处理...")
            result = agent_app.invoke(inputs)
            msgs = result.get("messages", [])
        except Exception as e:
            print(f"   ❌ Agent 执行发生异常/超时: {e}\n")
            continue
        
        # 1. 验证 Tool Calling 准确率
        used_tools = []
        for m in msgs:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict) and "name" in tc:
                        used_tools.append(tc["name"])
                    elif hasattr(tc, "name"):
                        used_tools.append(tc.name)
                
        is_tool_pass = item["expected_tool"] in used_tools
        if is_tool_pass:
            tool_correct += 1
            
        # 2. 验证生成文本关键词召回率（多关键词匹配）
        final_text = ""
        if msgs:
            ai_msgs = [m for m in msgs if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai"]
            target_msg = ai_msgs[-1] if ai_msgs else msgs[-1]
            final_text = extract_content_string(target_msg.content).lower()

        keywords = item.get("expected_keywords", [])
        matched_kw = [kw for kw in keywords if kw.lower() in final_text]
        is_kw_pass = len(matched_kw) > 0
        
        if is_kw_pass:
            keyword_hit += 1

        print(f"   ├─ Tool 匹配 ({item['expected_tool']}): {'✅' if is_tool_pass else '❌'} (实际调用: {used_tools})")
        print(f"   └─ 关键词召回 ({keywords}): {'✅' if is_kw_pass else '❌'} (命中关键词: {matched_kw})\n")

    print("=" * 40)
    print("📊 最终评估诊断报表：")
    print(f"• Tool Calling Accuracy: {tool_correct / total * 100:.2f}%")
    print(f"• Answer Relevance Rate: {keyword_hit / total * 100:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    run_comprehensive_eval()