import os
import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 原始数据目录与输出目录设置
RAW_TXT_DIR = BASE_DIR / "data" / "raw" / "dtc_codes"
RAW_CSV_DIR = BASE_DIR / "data" / "raw" / "mechanicdb"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_raw_txt_dtcs():
    """解析 37 个 txt DTC 文件"""
    dtc_docs = []
    if not RAW_TXT_DIR.exists():
        print(f"⚠️ 路径不存在: {RAW_TXT_DIR}")
        return dtc_docs

    for file_path in RAW_TXT_DIR.glob("*.txt"):
        brand = file_path.stem.replace("_codes", "").capitalize()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                match = re.match(r"^([A-Z][0-9A-Z]{3,5})\s*-\s*(.+)$", line, flags=re.IGNORECASE)
                if not match:
                    continue
                code = match.group(1).upper()
                desc = match.group(2).strip()

                content = (
                    f"故障码：\n{code}\n\n"
                    f"适用品牌：\n{brand}\n\n"
                    f"故障描述：\n{desc}\n\n"
                    f"数据来源：\nDTC Database\n\n"
                    f"类型：\nDTC Definition"
                )
                dtc_docs.append({
                    "content": content,
                    "metadata": {"dtc_code": code, "brand": brand, "doc_type": "dtc_definition"}
                })
    return dtc_docs

def process_mechanic_db():
    """解析 mechanicdb 路径下的 CSV 文件"""
    repair_docs = []
    parts_docs = []
    fix_to_dtc = {}

    # 1. 解析 diagnostic_fixes.csv 或 dtc_fixes_joined.csv
    fixes_csv = RAW_CSV_DIR / "dtc_fixes_joined.csv"
    if not fixes_csv.exists():
        fixes_csv = RAW_CSV_DIR / "diagnostic_fixes.csv"

    if fixes_csv.exists():
        with open(fixes_csv, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                code = row.get("dtc_code", "").strip().upper()
                fix_id = row.get("fix_id", "").strip()
                fault = row.get("detailed_technical_explanation", "").strip()
                solution = row.get("step_by_step_instructions", "").strip()
                difficulty = row.get("difficulty_level", "Standard").strip()
                hours = row.get("est_labor_hours", "N/A").strip()
                title = row.get("fix_title", "").strip()
                if not code or not fix_id:
                    continue
                fix_to_dtc[fix_id] = code

                content = (
                    f"故障码：\n{code}\n\n"
                    f"故障说明：\n{fault}\n\n"
                    f"维修项目：\n{title}\n\n"
                    f"维修方案：\n{solution}\n\n"
                    f"维修难度：\n{difficulty}\n\n"
                    f"预计工时：\n{hours}\n\n"
                    f"数据来源：\nMechanicDB\n\n"
                    f"类型：\nRepair Fix"
                )
                repair_docs.append({
                    "content": content,
                    "metadata": {"dtc_code": code, "fix_id": fix_id, "doc_type": "repair_fix"}
                })

    # 2. 解析 replacement_parts.csv
    parts_csv = RAW_CSV_DIR / "replacement_parts.csv"
    if parts_csv.exists():
        with open(parts_csv, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                fix_id = row.get("fix_id", "").strip()
                code = fix_to_dtc.get(fix_id, "")
                part_name = row.get("part_name", "").strip()
                if not code or not part_name:
                    continue

                content = (
                    f"故障码：\n{code}\n\n"
                    f"涉及配件：\n{part_name}\n\n"
                    f"关联维修编号：\n{fix_id}\n\n"
                    f"数据来源：\nMechanicDB\n\n"
                    f"类型：\nReplacement Parts"
                )
                parts_docs.append({
                    "content": content,
                    "metadata": {"dtc_code": code, "fix_id": fix_id, "doc_type": "replacement_parts"}
                })

    return repair_docs, parts_docs

def save_jsonl(docs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"✅ 已成功保存 {len(docs)} 条 Document 至 {output_path}")

if __name__ == "__main__":
    print("🚀 开始运行 Data Pipeline 规范化数据处理...")
    dtc_docs = process_raw_txt_dtcs()
    repair_docs, parts_docs = process_mechanic_db()

    save_jsonl(dtc_docs, PROCESSED_DIR / "dtc_documents.jsonl")
    save_jsonl(repair_docs, PROCESSED_DIR / "repair_documents.jsonl")
    save_jsonl(parts_docs, PROCESSED_DIR / "parts_documents.jsonl")
