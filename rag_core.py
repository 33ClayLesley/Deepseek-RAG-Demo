import tempfile
import os
import logging
import hashlib  # 引入哈希库
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from config import settings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        logger.info("初始化 RAG 引擎...")
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.vectorstore = None
        
        # 确保持久化目录存在
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    def _get_file_hash(self, file_bytes: bytes) -> str:
        """
        计算文件内容的 MD5 值。
        【升级】：把当前使用的 embedding_model 名字也掺进去。
        这样，如果换了模型，即便文件内容一样，也会生成新的集合，强制重新向量化。
        """
        # 将模型名称和文件内容拼接在一起计算 Hash
        # 需要先将 settings.embedding_model这一模型名encode成bytes 才能和file_bytes这个bytes类型拼接
        # 拼接顺序必须是 模型名 + 文件内容
        hash_input = settings.embedding_model.encode('utf-8') + file_bytes
        return hashlib.md5(hash_input).hexdigest()   # 最终返回32位十六进制字符串

    def process_pdf(self, uploaded_file) -> Chroma:
        """
        工程化升级：计算 Hash，如果已存在则直接加载，否则才解析入库
        """
        logger.info(f"开始处理 PDF 文件: {uploaded_file.name}")
        
        file_bytes = uploaded_file.getvalue()
        '''
        为什么_get_file_hash 中要拼 settings.embedding_model?
        不同的embedding模型，向量维度和语义空间都不同。换了模型之后，旧向量库中的向量和新模型查询出来的向量不在同个语义空间中
        直接混用会导致检索结果完全错乱。
        把模型名拼接进 hash 效果是：
        同一份PDF + 换了模型 -> hash变了 -> collection_name变了 -> 旧库不会被误用，系统自动建新库、重新向量化
        hash拼模型名 = 给向量库打上【模型指纹】 防止换模型后新旧向量混用 
        '''
        file_hash = self._get_file_hash(file_bytes)
        # 用 Hash 作为 collection_name，完美隔离不同文档，且自带去重属性
        collection_name = f"doc_{file_hash}"
        
        # 1. 尝试从本地持久化目录直接加载已有的库
        logger.info(f"检查本地是否已有该文档的索引: {collection_name}")
        # Chroma 不读任何文档 不调embedding 
        # embedding_function只是注册进去 告诉Chroma【以后你需要向量化文本时 用这个函数】
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir
        )

        logger.info(f"Chroma 初始化完成，当前 collection 条数: {self.vectorstore._collection.count()}")
        # 如果本地库里已经有数据了，直接返回（0秒响应）
        # _collection是Chroma的内部属性 不是公开API 也不是业界推荐用法
        # 生产代码最好用公开方法替代
        if self.vectorstore._collection.count() > 0:
            logger.info("命中历史缓存，直接加载已有索引，跳过解析与向量化。")
            return self.vectorstore

        # 2. 如果是全新文档，走完整的解析流程
        logger.info("未命中缓存，开始解析文档并生成向量...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splits = self.text_splitter.split_documents(docs)
            logger.info(f"文档切分完成，共 {len(splits)} 块。")

            # 将切分好的文档加入刚才初始化的 vectorstore 中
            # ！！！这里才真正调embedding --> add_documents
            # Chroma 把每个chunk的page_content传给embedding_function 拿到向量后 连同文本和metadata一起写入collection
            self.vectorstore.add_documents(splits)
            logger.info(f"入库完成，当前 collection 条数: {self.vectorstore._collection.count()}")
            logger.info(f"向量化完成，并已持久化到磁盘目录: {settings.chroma_persist_dir}")
            return self.vectorstore
        except Exception as e:
            logger.error(f"处理 PDF 时发生错误: {str(e)}")
            raise e
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.info(f"删除临时文件: {tmp_path}, 存在: {os.path.exists(tmp_path)}")


    def get_qa_chain(self):
        if not self.vectorstore:
            raise ValueError("Vectorstore 尚未初始化。")
            
        # 1. 向量检索器（语义召回）
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.retriever_fetch_k}
        )
        
        # 2. 获取所有文档用于构建 BM25 检索器（关键词精确召回）
        # 注意：在千万级数据中，BM25 应该存在 Elasticsearch 里
        # 这里为了教学演示，从 Chroma 提取到内存中
        results = self.vectorstore.get()
        # 需要把 Chroma 的格式转回 LangChain 的 Document
        docs = [Document(page_content=text, metadata=meta) 
                for text, meta in zip(results['documents'], results['metadatas'])]
        
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = settings.retriever_fetch_k
        
        # 3. 混合检索（Ensemble）：融合向量与 BM25，权重 5:5 开
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[1.0, 1.0]
        )
        
        # 4. 引入重排（Rerank）：把粗排召回的 20 条，用 Flashrank 筛选到 Top 3
        compressor = FlashrankRerank(top_n=settings.retriever_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        
        # 修改 Prompt，加入 chat_history 的占位符
        # {context} 的内容是create_retrieval_chain 在内部把检索器返回的 documents 自动塞进去的
        system_prompt = (
            "你是一个严谨的文档分析助手。请严格根据以下【上下文】和【历史对话】回答用户的问题。\n"
            "规则：\n"
            "1. 如果【上下文】中找不到答案，请直接回答“根据当前文档，无法找到相关信息”，绝不允许编造。\n"
            "2. 回答时请保持客观、准确，并在引用的句子末尾加上来源角标，例如 [1]、[2]。\n"
            "3. 不要复述问题，直接给出答案。\n\n"
            "【上下文】\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            # 系统信息 放强约束规则和{context} 每次调用都固定在最前面 优先级最高
            ("system", system_prompt),
            # 插入历史对话记录 调用时传入 chay_history 列表
            MessagesPlaceholder(variable_name="chat_history"),
            # 当前用户问题 {input}由调用方传入
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
        
        return rag_chain

    def stream_chat(self, prompt: str, history: list):
        """流式生成回答，并同时返回文档上下文"""
        qa_chain = self.get_qa_chain()
    
        chat_history = []
        for msg in history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
    
        final_context = []

        '''
        为什么 stream 能同时拿到 context 和 answer?
        因为 create_retrieval_chain 流式输出是分阶段yield的
        先执行 retriever 拿到documents --> yield一个包含"context"的chunk
        --> 再把context+input传给LLM LLM流式生成 --> 逐段yield包含"answer"的chunk
        '''
        for chunk in qa_chain.stream({
            "input": prompt,
            "chat_history": chat_history
        }):
            if "context" in chunk:
                final_context = chunk["context"]
                yield {"type": "context", "content": final_context}
            if "answer" in chunk:
                yield {"type": "text", "content": chunk["answer"]}