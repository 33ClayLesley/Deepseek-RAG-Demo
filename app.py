import streamlit as st
import logging
from langchain_openai import ChatOpenAI
from rag_core import RAGEngine
from config import settings

# ==========================================
# 1. 配置日志系统：屏蔽底层 HTTP 噪音，只保留业务日志
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

st.set_page_config(page_title="你的文档解读AI助手", layout="wide")
st.title("🤖 智能文档问答助手 (DeepSeek + RAG)")

# cache_resource 让引擎只初始化一次 全局复用
@st.cache_resource(show_spinner=False)
def get_rag_engine():
    return RAGEngine()
engine = get_rag_engine()

with st.sidebar:
    st.markdown("### 🔑 配置")
    deepseek_api_key = st.text_input("DeepSeek API Key (留空则使用 .env 配置)", type="password")
    st.markdown("---")
    st.markdown("### 📂 上传文档")
    uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf")
    
    # ==========================================
    # 2. 新增：强制重新处理按钮
    # ==========================================
    if st.button("🔄 重新解析当前文档"):
        if "processed_file_sig" in st.session_state:
            del st.session_state.processed_file_sig
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 3. 优化：文件更新判断逻辑，引入文件大小作为签名
# ==========================================
if uploaded_file:
    # 使用文件名 + 大小 作为唯一签名，防止同名不同内容被忽略
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if st.session_state.get("processed_file_sig") != file_signature:
        with st.spinner(f"正在分析文档《{uploaded_file.name}》..."):
            try:
                if deepseek_api_key:
                    engine.llm = ChatOpenAI(
                        model=settings.llm_model,
                        api_key=deepseek_api_key,
                        base_url=settings.llm_base_url,
                        temperature=settings.llm_temperature
                    )

                engine.process_pdf(uploaded_file)
                st.session_state.processed_file_sig = file_signature
                st.session_state.messages = [] 
                st.success(f"文档《{uploaded_file.name}》处理完成！")
            except Exception as e:
                st.error(f"处理文档时出错: {e}")

# ==========================================
# 4. 问答交互逻辑
# ==========================================
if prompt := st.chat_input("想问什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if engine.vectorstore:
        # SecretStr 要拿到真实值必须显式调用 .get_secret_value()\
        # 因此 用SecretStr可以防止Key意外泄露到日志或异常堆栈
        current_key = engine.llm.openai_api_key.get_secret_value()
        if deepseek_api_key and current_key != deepseek_api_key:
             engine.llm = ChatOpenAI(
                 model=settings.llm_model,
                 api_key=deepseek_api_key,
                 base_url=settings.llm_base_url,
                 temperature=settings.llm_temperature
             )

        with st.chat_message("assistant"):
            try:
                stream_gen = engine.stream_chat(
                    prompt=prompt, 
                    history=st.session_state.messages[:-1]
                )
                
                def text_generator():
                    for chunk in stream_gen:
                        if chunk["type"] == "text":
                            yield chunk["content"]
                        elif chunk["type"] == "context":
                            st.session_state.current_context = chunk["content"]
                            
                answer = st.write_stream(text_generator())
                
                if hasattr(st.session_state, "current_context"):
                    with st.expander("🔍 查看原文出处"):
                        for i, doc in enumerate(st.session_state.current_context):
                            page = doc.metadata.get("page", "未知")
                            page_display = page + 1 if isinstance(page, int) else page
                            st.markdown(f"**[{i+1}] (第 {page_display} 页):** {doc.page_content[:150]}...")
                    del st.session_state.current_context
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"调用出错: {e}")
    else:
        st.warning("请先上传并处理 PDF 文件！")