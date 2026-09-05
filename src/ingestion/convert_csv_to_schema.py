import os
import json
import re

PROCESSED_DIR = "data/processed"
RAW_DTC_DIR = "data/raw/dtc_codes"
RAW_MECHANIC_DIR = "data/raw/mechanicdb"
OUTPUT_FILE = "data/structured_dtc_database.json"

def extract_dtc_code(text: str) -> str:
    """提取标准 DTC 编码（如 P0301）"""
    match = re.search(r'[PBCU]\d{4}', text.upper())
    return match.group(0) if match else ""

def load_jsonl(filepath: str) -> list:
    """按行读取 JSONL 文件"""
    items = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
    return items

def process_and_merge_all():
    print("🔄 开始整合数据源 (Processed JSONL + Raw txt/csv)...")
    records_dict = {}

    # 1. 优先解析 processed 目录下的 jsonl 文件
    dtc_docs = load_jsonl(os.path.join(PROCESSED_DIR, "dtc_documents.jsonl"))
    repair_docs = load_jsonl(os.path.join(PROCESSED_DIR, "repair_documents.jsonl"))
    parts_docs = load_jsonl(os.path.join(PROCESSED_DIR, "parts_documents.jsonl"))

    for doc in dtc_docs:
        content = doc.get("page_content", doc.get("content", ""))
        metadata = doc.get("metadata", {})
        code = extract_dtc_code(metadata.get("dtc", content))
        if not code:
            continue
            
        records_dict[code] = {
            "dtc": code,
            "system": metadata.get("system", "Engine"),
            "component": metadata.get("component", "General Component"),
            "description": metadata.get("description", content[:100]),
            "symptoms": metadata.get("symptoms", ["Engine fault light on"]),
            "possible_causes": metadata.get("possible_causes", ["Component failure"]),
            "diagnostic_steps": [],
            "repair": [],
            "parts": [],
            "verification": {
                "drive_cycle": "Clear DTC, idle for 5 mins, and conduct road test",
                "clear_dtc": True
            }
        }

    # 2. 补全 repair_documents.jsonl 维修步骤
    for doc in repair_docs:
        content = doc.get("page_content", doc.get("content", ""))
        code = extract_dtc_code(content)
        if code in records_dict:
            step_idx = len(records_dict[code]["diagnostic_steps"]) + 1
            records_dict[code]["diagnostic_steps"].append({
                "step": step_idx,
                "test": content.strip(),
                "measurement": "Signal / Resistance check",
                "expected": "Within standard tolerance range"
            })
            records_dict[code]["repair"].append(content.strip())

    # 3. 补全 parts_documents.jsonl 备件数据
    for doc in parts_docs:
        content = doc.get("page_content", doc.get("content", ""))
        code = extract_dtc_code(content)
        if code in records_dict:
            records_dict[code]["parts"].append({
                "part_number": "OEM Standard",
                "component": content.strip()
            })

    # 4. 补充扫描 raw/dtc_codes/ 文本（作为兜底索引）
    if os.path.exists(RAW_DTC_DIR):
        for fname in os.listdir(RAW_DTC_DIR):
            if fname.endswith(".txt"):
                brand = fname.replace("_codes.txt", "").capitalize()
                fpath = os.path.join(RAW_DTC_DIR, fname)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        code = extract_dtc_code(line)
                        if code and code not in records_dict:
                            records_dict[code] = {
                                "dtc": code,
                                "system": "Engine",
                                "component": f"{brand} Engine System",
                                "description": line,
                                "symptoms": ["Check Engine Light"],
                                "possible_causes": ["System Circuit / Sensor Error"],
                                "diagnostic_steps": [
                                    {
                                        "step": 1,
                                        "test": f"Inspect {code} sensor connection and harness",
                                        "measurement": "Voltage / Continuity",
                                        "expected": "Normal"
                                    }
                                ],
                                "repair": [f"Replace {code} faulty sensor or repair harness"],
                                "parts": [{"part_number": "OEM Standard", "component": "Replacement Sensor"}],
                                "verification": {"drive_cycle": "Drive cycle self-check", "clear_dtc": True}
                            }

    # 5. 输出融合成果
    final_data = list(records_dict.values())
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"🎉 处理完成！成功融合 {len(final_data)} 条全量深层 Schema 数据到 {OUTPUT_FILE}")

if __name__ == "__main__":
    process_and_merge_all()