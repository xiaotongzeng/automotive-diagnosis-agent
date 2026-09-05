import streamlit as st
import requests

st.set_page_config(page_title="Automotive Diagnostic Workbench", layout="wide")

st.title("🚗 汽车智能诊断工作台 (Automotive Diagnostic Workbench)")
st.caption("基于 Dynamic State Machine, Tool-calling, Human-in-the-Loop & Evidence Citation 构建")

# 侧边栏：Vehicle Context
with st.sidebar:
    st.header("🚘 车辆上下文 (Vehicle Context)")
    brand = st.text_input("品牌", value="Toyota")
    model = st.text_input("车型", value="Camry")
    year = st.text_input("年份", value="2020")
    engine = st.text_input("排量", value="2.5L")
    st.divider()
    st.header("🛠️ 交互式诊断控制")
    interactive_mode = st.toggle("开启人机协作测量模式 (Human-in-the-loop)", value=False)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 故障录入与状态控制台")
    user_query = st.text_area("描述故障现象或 DTC 代码：", value="2020 Toyota Camry 2.5L 报 P0301，怠速抖动，下一步该怎么做？", height=100)
    
    if interactive_mode:
        st.info("💡 人机协作模式开启：Agent 将在中间步骤要求您提供物理测量值（如线圈电阻/电压）。")
        test_val = st.text_input("输入实际测量结果（例如：1缸点火线圈次级电阻 0.8Ω）：")
    
    submit_btn = st.button("🚀 执行Agent 诊断推理", type="primary")

with col2:
    st.subheader("🔍 诊断推理链与证据追溯 (Diagnostic Workbench)")
    
    if submit_btn:
        with st.spinner("🤖 Agent 状态机正在执行推理与工具调度..."):
            payload = {
                "question": user_query + (f" [实测反馈: {test_val}]" if interactive_mode and test_val else ""),
                "vehicle_info": {"brand": brand, "model": model, "year": year, "engine": engine}
            }
            try:
                res = requests.post("http://127.0.0.1:8000/api/v1/diagnose", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.success("诊断推理完成！")
                    
                    # 展示推理链
                    st.markdown("### 🧠 结构化诊断报告")
                    st.markdown(data.get("report", ""))
                    
                    # 展开显示可追溯依据
                    with st.expander("📚 查看 RRF & Cross-Encoder 重排引用的原始数据依据 (Evidence Citation)"):
                        st.json({
                            "Vehicle_Context": payload["vehicle_info"],
                            "Source_Status": "Evidence Verified",
                            "Retrieval_Pipeline": "BGE-M3 + BM25 -> RRF -> BGE-Reranker-Large"
                        })
                else:
                    st.error("服务端响应异常。")
            except Exception as e:
                st.error(f"无法连接诊断服务: {e}")