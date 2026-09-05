# Automotive Diagnosis Agent

基于 **RAG（Retrieval-Augmented Generation）** 与 **LangGraph Agent 工作流**构建的汽车故障智能诊断系统。

本项目面向汽车故障诊断场景，将故障码知识、维修方案和零件信息构建为可检索知识库，并通过混合检索、重排序以及 Agent 工具调用，为用户提供结构化的故障分析与维修建议。

---

## 1. 项目简介

传统汽车故障诊断通常依赖维修手册、故障码数据库和维修经验，需要人工查询大量资料。

本项目通过 **RAG + Agent** 的方式，将汽车故障诊断知识与大语言模型结合，实现从用户故障描述到知识检索、证据分析和诊断结果生成的自动化流程。

项目主要实现：

* 汽车 DTC（Diagnostic Trouble Code）故障码知识库
* 故障码结构化信息检索
* 维修方案检索
* 汽车零件信息检索
* BM25 + 向量检索的混合检索
* RRF（Reciprocal Rank Fusion）结果融合
* BGE-Reranker-Large 重排序
* 基于 LangGraph 的 Agent 工作流
* FastAPI API 服务
* Streamlit 可视化交互界面
* 自动化诊断效果评估

---

## 2. 系统架构

系统整体采用以下处理流程：

```text
用户输入汽车故障信息
        │
        ▼
┌─────────────────────┐
│  Automotive Agent   │
│    LangGraph        │
└─────────┬───────────┘
          │
          ▼
    判断是否需要工具
          │
     ┌────┴────┐
     │         │
     ▼         ▼
  Agent     Tool Call
               │
               ▼
      ┌──────────────────┐
      │ Hybrid Retriever │
      └────────┬─────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
  BM25 Retrieval   Vector Retrieval
       │                │
       └───────┬────────┘
               ▼
        RRF Result Fusion
               │
               ▼
      BGE-Reranker-Large
               │
               ▼
        Evidence Retrieval
               │
               ▼
        Agent Diagnosis
               │
               ▼
          Final Report
```

---

## 3. 核心功能

### 3.1 DTC 故障码检索

系统支持根据汽车故障码检索相关诊断信息。

例如：

```text
P0301
P1106
P1203
```

Agent 可以调用：

```text
lookup_dtc_schema
```

获取相关故障码信息。

---

### 3.2 维修方案检索

根据故障码和故障描述，从维修知识库中检索相关维修步骤。

Agent 工具：

```text
search_repair_procedure
```

---

### 3.3 零件信息检索

根据故障码和诊断场景检索可能涉及的汽车零部件。

Agent 工具：

```text
search_parts
```

---

### 3.4 混合检索

项目采用：

```text
BM25
+
Vector Retrieval
```

进行混合检索。

其中：

* BM25：适合精确匹配故障码、专业术语和关键词
* 向量检索：适合语义相似度检索
* RRF：融合不同检索方式的结果

最终再使用：

```text
BGE-Reranker-Large
```

对候选结果进行重排序。

---

## 4. 技术栈

| 模块                | 技术                               |
| ----------------- | -------------------------------- |
| LLM               | DeepSeek / OpenAI-compatible API |
| Agent             | LangGraph                        |
| LLM Framework     | LangChain                        |
| Embedding         | BAAI/bge-m3                      |
| Reranker          | BAAI/bge-reranker-large          |
| Vector Database   | Chroma                           |
| Keyword Retrieval | BM25                             |
| API               | FastAPI                          |
| Web UI            | Streamlit                        |
| Language          | Python                           |

---

## 5. 项目目录

```text
CAR_RAG_AGENT/
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data_pipeline/
│   ├── process_data.py
│   └── build_vector_store.py
│
├── evaluation/
│   └── evaluate.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── tools.py
│   │   └── workflow.py
│   │
│   └── ingestion/
│       ├── __init__.py
│       └── hybrid_retriever.py
│
├── api_server.py
├── main.py
├── ui_app.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. 环境要求

建议使用：

* Python 3.10+
* Git
* 具有网络访问能力的环境

部分 Embedding 和 Reranker 模型首次运行时需要从 Hugging Face 下载。

---

## 7. 安装项目

克隆项目：

```bash
git clone https://github.com/xiaotongzeng/automotive-diagnosis-agent.git
cd automotive-diagnosis-agent
```

创建虚拟环境：

```bash
python3 -m venv .venv
```

激活虚拟环境：

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 8. 配置环境变量

项目提供：

```text
.env.example
```

复制为：

```bash
cp .env.example .env
```

然后根据实际使用的模型服务填写 API Key。

示例：

```text
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
DASHSCOPE_API_KEY=

OPENAI_API_BASE=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat

EMBEDDING_MODEL_NAME=BAAI/bge-m3
RERANKER_MODEL_NAME=BAAI/bge-reranker-large

CAR_RAG_COLLECTION=car_dtc_collection_v2
```

**不要将真实 API Key 提交到 GitHub。**

---

## 9. 数据准备

项目的数据处理流程位于：

```text
data_pipeline/
```

原始数据默认放置于：

```text
data/raw/
```

其中 DTC 数据目录为：

```text
data/raw/dtc_codes/
```

维修相关数据位于：

```text
data/raw/mechanicdb/
```

---

## 10. 数据处理

运行：

```bash
python3 data_pipeline/process_data.py
```

程序会对原始数据进行处理，并生成：

```text
data/processed/
├── dtc_documents.jsonl
├── repair_documents.jsonl
└── parts_documents.jsonl
```

这些文件作为后续知识库构建的数据来源。

---

## 11. 构建向量知识库

项目使用 Chroma 保存向量数据。

运行：

```bash
python3 data_pipeline/build_vector_store.py
```

如果需要重新构建：

```bash
python3 data_pipeline/build_vector_store.py --reset
```

程序使用：

```text
BAAI/bge-m3
```

生成文本 Embedding，并建立 Chroma 向量索引。

同时可以建立 BM25 索引，为混合检索提供支持。

---

## 12. Agent 工作流

核心 Agent 位于：

```text
src/agent/workflow.py
```

Agent 使用 LangGraph 构建。

核心节点包括：

```text
agent
call_tool
```

基本流程：

```text
用户问题
   │
   ▼
Agent
   │
   ├── 不需要工具 ──► 最终回答
   │
   └── 需要工具
          │
          ▼
       Tool Call
          │
          ▼
        Agent
```

Agent 根据用户输入决定是否调用知识库工具。

---

## 13. Agent 工具

目前提供三个主要工具：

### DTC 故障码查询

```text
lookup_dtc_schema(code)
```

### 维修方案查询

```text
search_repair_procedure(code)
```

### 零件查询

```text
search_parts(code)
```

这些工具负责从汽车诊断知识库中获取相关证据。

---

## 14. 运行 API 服务

项目提供 FastAPI 服务。

运行：

```bash
python3 api_server.py
```

默认服务地址：

```text
http://127.0.0.1:8000
```

诊断接口：

```text
POST /api/v1/diagnose
```

请求示例：

```json
{
  "question": "车辆出现 P0301 故障码并且怠速抖动，应该如何检查？",
  "vehicle_info": {
    "brand": "Toyota",
    "model": "Camry",
    "year": 2020,
    "engine": "2.5L"
  }
}
```

API 会返回诊断结果以及车辆上下文信息。

---

## 15. 运行 Streamlit Web UI

项目提供可视化诊断界面：

```text
ui_app.py
```

首先启动 API：

```bash
python3 api_server.py
```

然后在另一个终端运行：

```bash
streamlit run ui_app.py
```

浏览器打开 Streamlit 提供的本地地址即可使用。

Web UI 支持：

* 车辆品牌
* 车辆型号
* 车辆年份
* 发动机信息
* 故障描述
* Human-in-the-loop 输入
* AI 诊断结果展示
* 检索流程信息展示

---

## 16. 诊断检索流程

项目当前的核心检索链路为：

```text
用户问题
   │
   ▼
BM25 + Vector Retrieval
   │
   ▼
RRF Fusion
   │
   ▼
BGE-Reranker-Large
   │
   ▼
Top-K Evidence
   │
   ▼
LLM / Agent
   │
   ▼
Diagnosis Report
```

其中：

```text
BGE-M3 + BM25
        ↓
       RRF
        ↓
BGE-Reranker-Large
```

用于提升汽车故障诊断知识检索的相关性。

---

## 17. 评估

项目提供自动化评估程序：

```text
evaluation/evaluate.py
```

运行：

```bash
python3 evaluation/evaluate.py
```

当前评估覆盖：

* DTC 精确查询
* 根据车辆上下文进行维修方案检索
* 零件信息查询

评估程序会检查 Agent 是否正确调用相应工具以及返回结果中是否包含预期信息。

---

## 18. Python 代码检查

在提交代码之前，可以运行：

```bash
python3 -m compileall -q .
```

如果命令没有输出，通常表示 Python 文件通过了基本语法编译检查。

---

## 19. 常见问题

### 19.1 API Key 未配置

如果运行 Agent 时提示 API Key 缺失，请检查：

```text
.env
```

并确认已经配置对应模型服务的 API Key。

---

### 19.2 Embedding 模型下载失败

项目首次建立向量库时需要下载：

```text
BAAI/bge-m3
```

如果网络环境无法正常访问模型仓库，需要解决模型下载或网络访问问题后再运行。

---

### 19.3 Reranker 模型下载失败

项目使用：

```text
BAAI/bge-reranker-large
```

首次运行检索流程时可能需要下载模型。

---

### 19.4 向量库不存在

如果项目没有找到 Chroma 向量库，需要先执行：

```bash
python3 data_pipeline/build_vector_store.py
```

---

## 20. 安全说明

请勿将以下内容提交到 GitHub：

```text
.env
API Keys
Token
密码
本地数据库
模型权重
大型数据文件
```

项目通过：

```text
.env.example
```

提供环境变量配置模板。

真实的 `.env` 文件应保持在本地，并由 `.gitignore` 忽略。

---

## 21. 项目当前状态

当前项目已经实现：

* [x] 汽车 DTC 知识检索
* [x] 维修方案检索
* [x] 零件信息检索
* [x] BM25 检索
* [x] 向量检索
* [x] RRF 融合
* [x] BGE-Reranker 重排序
* [x] LangGraph Agent
* [x] FastAPI API
* [x] Streamlit UI
* [x] 自动化评估

后续可以继续扩展：

* 更多车型与维修数据
* 更完整的维修手册知识库
* 多轮诊断对话
* 诊断证据可视化
* 更完善的评估数据集
* 故障诊断路径优化
* 多模态汽车故障诊断

---

## 22. License

当前仓库暂未提供正式的 LICENSE 文件。

如用于开源发布，建议根据项目实际使用的数据、代码和模型许可情况补充对应 License。

