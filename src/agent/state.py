from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class DiagnosticState(TypedDict):
    # 基础对话消息
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 车辆与输入解析上下文
    vehicle_info: Dict[str, str]  # brand, model, year, engine
    dtc_codes: List[str]
    symptoms: List[str]
    
    # 证据链追溯
    retrieved_evidences: List[Dict[str, Any]]  # 包含 text, source, rerank_score 等
    
    # 推理与诊断链
    diagnostic_hypotheses: List[Dict[str, Any]]
    current_step: str
    test_results: List[Dict[str, str]]  # 用户反馈的实测结果
    confidence: float
    
    # 证据是否充分控制标记
    evidence_sufficient: bool
    human_input_needed: bool