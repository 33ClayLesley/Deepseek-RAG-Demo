# 🤖 基于 RAG 的智能文档问答助手 (DeepSeek 版)

这是一个基于 **Streamlit**、**LangChain** 和 **DeepSeek** 构建的生产级 RAG (检索增强生成) 应用。允许用户上传 PDF 文档，通过混合检索、重排序与强约束 Prompt，实现高准确度、防幻觉、可溯源的问答体验。

 ![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-v1.2-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.52-yellow) ![DeepSeek](https://img.shields.io/badge/Model-DeepSeek--V3-orange)

## 🌠 项目亮点（工程级特性）

- **⚡ 混合检索与精排 (Hybrid Search + Rerank)**：抛弃单一的向量相似度，采用「BM25 关键词检索 + 向量检索」双路召回（粗排 Top 20），并通过 `Flashrank` 模型进行二次重排（精排 Top 3），彻底解决专有名词漏召和噪音过载问题。
- **💾 资产持久化与秒级缓存**：告别内存模式，向量库落盘至本地 `chroma_db`。引入 **MD5 Hash 去重机制**，同一文档二次加载 0 秒响应，重启不丢失任何计算成果。
- **🛡️ 强约束 Prompt 防幻觉**：摒弃默认模板，自定义 System Prompt 强制要求大模型“仅依据上下文回答，未知则回答不知道”，并强制输出 `[1]、[2]` 引用角标。
- **🌊 流式输出与多轮对话**：采用 LCEL 架构重构生成链，支持打字机流式输出（SSE）。在单次检索中同时捕获文本流与上下文出处，拒绝双调 API，大幅降低成本。
- **🧪 完善的单元测试**：利用 `pytest` + `mocker` 实现了核心逻辑的隔离测试（包括缓存命中、解析异常、空文件处理），具备 CI/CD 落地的基础。

## 🛠️ 技术栈

- **前端交互**：Streamlit
- **应用编排**：LangChain (Core + Community + Classic) / LCEL
- **大模型**：DeepSeek Chat (兼容 OpenAI 协议)
- **向量数据库**：ChromaDB (Local Persistent 模式)
- **Embedding**：HuggingFace 本地模型（默认：`BAAI/bge-small-zh-v1.5`，中文语义优化）
- **检索增强**：`rank_bm25` + `flashrank` (Cross-Encoder 重排)
- **配置管理**：Pydantic-Settings + `.env`

## 📂 项目结构

```text
.
├── app.py                 # Streamlit 前端界面
├── rag_core.py            # RAG 核心引擎 (类的封装)
├── config.py              # Pydantic 配置管理
├── requirements.txt       # 锁定版本的依赖清单
├── .env                   # 环境变量 (需自行创建)
├── .dockerignore          # Docker 忽略文件
├── Dockerfile             # 容器化部署文件
├── chroma_db/             # 向量数据库持久化目录 (自动生成)
└── tests/                 # 单元测试目录
    └── test_rag_core.py
```

## 🚀快速开始

### 1.申请Deepseek的API key

请前往Deepseek开放注册平台申请API key，sk开头的序列号(申请创建完成后记得先复制保存下来！)

### 2.安装依赖


```python
# 创建并激活虚拟环境 (可选)
python -m venv .venv
source .venv/bin/activate  # Windows 下: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```


### 3.配置环境变量
```python
LLM_API_KEY=sk-你的API_KEY
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CHROMA_PERSIST_DIR=./chroma_db
```

### 4.运行测试
```python
pytest tests/ -v
```

### 5.启动应用
```python
streamlit run app.py
```

## 📖 使用指南

1.打开浏览器访问 ```http://localhost:8501```。

2.在左侧边栏上传 PDF 文件（首次运行会自动下载 Embedding 模型至本地）。

3.系统会自动解析、切分、向量化并持久化存储，出现“处理完成”提示后即可在对话框提问。

4.在回答下方点击「查看原文出处」，可直接溯源至具体页码及文本切片。

## ⚠️常见问题

- 报错 `ModuleNotFoundError: No module named 'langchain.chains'`
  - **原因**：LangChain v1.0+ 进行了包拆分，移除了内置的旧版 Chain。
  - **解决**：项目已彻底迁移至 ```langchain-classic```和 LCEL 架构，请确保通过 ```requirements.txt```完整安装依赖。

- 报错 `Database is locked` (Windows)
  - **原因**：ChromaDB 在持久化模式下，多线程同时读写引发的文件锁。
  - **解决**：项目已使用唯一```collection_name```隔离不同文档，且加入了单例缓存机制，通常不会报错。如遇报错，请确保在本地只启动了一个 Streamlit 实例。

- 想替换 Embedding 模型？
  - **解决**：修改```.env```中的 ```EMBEDDING_MODEL```即可。由于引入了 Hash 去重，更换模型后系统会自动识别为新集合，不会导致数据损坏或冲突，但旧模型生成的数据需要删除```chroma_db```目录重新生成。



*Create by Lesley. Powered by LangChain & DeepSeek.*