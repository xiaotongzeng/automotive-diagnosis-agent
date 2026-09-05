import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 1. 严格定义 Pydantic Schema
class DiagnosticStep(BaseModel):
    step: int
    test: str
    measurement: Optional[str] = None
    expected: Optional[str] = None

class PartItem(BaseModel):
    part_number: Optional[str] = None
    component: str

class Verification(BaseModel):
    drive_cycle: Optional[str] = None
    clear_dtc: bool = True

class StandardDTCRecord(BaseModel):
    dtc: str
    system: str = "Engine"
    component: str
    description: str
    symptoms: List[str] = []
    possible_causes: List[str] = []
    diagnostic_steps: List[DiagnosticStep] = []
    repair: List[str] = []
    parts: List[PartItem] = []
    verification: Optional[Verification] = None

    def to_rag_document_text(self) -> str:
        """转化为适用于 Vector Store / BM25 索引的深层富文本表达"""
        steps_str = "\n".join([
            f"  步骤 {s.step}: {s.test} | 测试项: {s.measurement or '外观/信号'} | 标准范围: {s.expected or '正常'}"
            for s in self.diagnostic_steps
        ])
        parts_str = ", ".join([f"{p.component} (OEM: {p.part_number or '通用'})" for p in self.parts])
        
        return f"""[故障码定义] {self.dtc} - {self.description} (所属系统: {self.system}, 部件: {self.component})
[常见故障现象] {', '.join(self.symptoms)}
[可能根因分析] {', '.join(self.possible_causes)}
[标准化诊断排查链与检测公差]
{steps_str}
[建议维修方案] {', '.join(self.repair)}
[适配备件清单] {parts_str}
[维修后验证程序] 清除DTC: {self.verification.clear_dtc if self.verification else True} | Drive Cycle: {self.verification.drive_cycle if self.verification else '运行工况自检'}"""