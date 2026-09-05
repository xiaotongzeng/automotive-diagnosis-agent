# Automotive Diagnosis Agent

基于 RAG（Retrieval-Augmented Generation）与 Agent 工作流构建的汽车故障智能诊断系统。

## 1. 项目简介

本项目面向汽车故障诊断场景，结合知识库检索、LLM 推理与 Agent 工作流，实现从用户故障描述到相关知识检索、故障分析和诊断建议生成的完整流程。

项目主要目标：

- 构建汽车故障诊断领域知识库
- 实现基于 RAG 的知识检索
- 构建汽车故障诊断 Agent
- 支持多步骤诊断推理
- 提供 API 服务
- 提供可视化交互界面
- 建立自动化评估体系

## 2. 项目架构

```text
CAR_RAG_AGENT
│
├── config/             # 项目配置
├── data_pipeline/      # 数据处理与知识库构建
├── evaluation/         # RAG / Agent 评估
├── src/                # 核心业务代码
│
├── api_server.py       # API 服务入口
├── main.py             # Agent 主入口
├── ui_app.py           # Web UI
│
├── .env.example        # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md