import os
import sys
from pathlib import Path
from typing import Dict, Any, List, TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 1. 导入配置变量
from config.settings import DEEPSEEK_API_KEY, LLM_MODEL_NAME
try:
    from config.settings import QWEN_BASE_URL as BASE_URL
except ImportError:
    from config.settings import OPENAI_API_BASE as BASE_URL

# 2. 动态兼容导入 tools
import src.agent.tools as tools_module

if hasattr(tools_module, "agent_tools"):
    TOOLS = tools_module.agent_tools
elif hasattr(tools_module, "tools"):
    TOOLS = tools_module.tools
else:
    # 如果 tools.py 导出的是独立函数
    TOOLS = [
        getattr(tools_module, attr) 
        for attr in dir(tools_module) 
        if not attr.startswith("_") and callable(getattr(tools_module, attr))
    ]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

# ----------------------------------------------------------------------
# 3. 初始化 LLM（配置 15 秒超时，防卡死）
# ----------------------------------------------------------------------
llm = ChatOpenAI(
    model=LLM_MODEL_NAME if LLM_MODEL_NAME else "deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL if BASE_URL else "https://api.deepseek.com",
    request_timeout=15.0,
    max_retries=2
).bind_tools(TOOLS)

# ----------------------------------------------------------------------
# 4. 定义 State 状态结构
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    vehicle_info: Dict[str, Any]

# ----------------------------------------------------------------------
# 5. Agent 核心图节点逻辑
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个专业的汽车故障诊断与维修 AI 助手。
你可以使用给定的工具查找故障码（DTC）、检索维修流程以及查询所需零部件。
请结合用户提供的车辆信息（品牌、车型、年份、发动机）给出精准、简明的诊断与排查建议。"""

def agent_node(state: AgentState) -> Dict[str, Any]:
    messages = list(state.get("messages", []))
    vehicle_info = state.get("vehicle_info", {})
    
    sys_content = SYSTEM_PROMPT
    if vehicle_info:
        sys_content += f"\n当前车辆信息: {vehicle_info}"
    
    full_messages = [SystemMessage(content=sys_content)] + messages
    response = llm.invoke(full_messages)
    return {"messages": [response]}

def tool_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    tool_outputs = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")
            
            tool_fn = TOOLS_BY_NAME.get(tool_name)
            if tool_fn:
                try:
                    res = tool_fn.invoke(tool_args)
                    res_str = str(res)
                except Exception as err:
                    res_str = f"Tool 执行发生错误: {err}"
            else:
                res_str = f"未找到名为 {tool_name} 的工具"
                
            tool_outputs.append(ToolMessage(content=res_str, tool_call_id=tool_id, name=tool_name))
            
    return {"messages": tool_outputs}

def should_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    return END

# ----------------------------------------------------------------------
# 6. 构建与编译 LangGraph 工作流
# ----------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("call_tool", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "call_tool",
        END: END
    }
)

workflow.add_edge("call_tool", "agent")

app = workflow.compile()