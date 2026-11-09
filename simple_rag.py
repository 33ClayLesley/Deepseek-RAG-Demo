import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import shutil

load_dotenv()

def load_pdf(path):
    """PyPDFLoader 加载，返回 List[Document]，打印文档数和第一页前 100 字"""
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f"文档数: {len(docs)}")
    print(f"第一页前100字: {docs[0].page_content[:100]}")
    return docs


def split_docs(docs):
    """RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    打印 chunk 数和前 2 个 chunk 内容"""
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    splits = splitter.split_documents(docs)
    print(f"chunk 数: {len(splits)}")
    print(f"chunk 0: {splits[0].page_content[:100]}")
    return splits

def build_vectorstore(splits):
    if os.path.exists("./simple_chroma_db"):
        shutil.rmtree("./simple_chroma_db")

    """HuggingFaceEmbeddings(BAAI/bge-small-zh-v1.5) + Chroma
    打印 collection 条数"""
    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma.from_documents(
        documents = splits,
        embedding = embeddings,
        persist_directory = "./simple_chroma_db"
    )
    print(f"入库条数: {vectorstore._collection.count()}")
    return vectorstore


def build_chain(vectorstore):
    """retriever(k=3) + 强约束 prompt + ChatOpenAI + create_stuff_documents_chain + create_retrieval_chain"""
    # 设置一个检索器 最终召回top3
    # retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})

    # 加回BM25 + Rerank
    # 1、向量检索器-粗排20
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    # 2、BM25检索器-粗排20
    '''
    BM25需要先把Chroma里的全量文档拉回内存 才能建立关键词索引
    没有vectorStore.get() BM25Retriever.from_documents就没数据可建
    '''
    results = vectorstore.get()
    docs = [Document(page_content=text, metadata=meta)
            for text, meta in zip(results['documents'], results['metadatas'])]

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 20

    # 3、Ensemble融合 + Rerank精排3
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights = [0.5, 0.5]
    )

    compressor = FlashrankRerank(top_n = 3)
    '''
    换了retriever的实现 上层链条不变
    '''
    retriever = ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever = ensemble_retriever
    )

    # 设置系统级prompt 防止模型幻觉、回答冗余、没有溯源
    system_prompt = (
        "你是一个严谨的文档分析助手。请严格根据以下【上下文】回答用户的问题。\n"
        "规则：\n"
        "1. 如果【上下文】中找不到答案，请直接回答“根据当前文档，无法找到相关信息”，绝不允许编造。\n"
        "2. 回答时在引用的句子末尾加上来源角标，例如 [1]、[2]。\n"
        "3. 不要复述问题，直接给出答案。\n\n"
        "【上下文】\n"
        "{context}"
    )

    # 只接受一个列表参数，输入基于系统级prompt加入当前用户提问的问题
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 调用大模型 模型是Deepseek 用os.getenv显式调用API 设置temperature为0.1防止乱答
    llm = ChatOpenAI(
        model = "deepseek-chat",
        api_key = os.getenv("LLM_API_KEY"),
        base_url = "https://api.deepseek.com",
        temperature = 0.1
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    # create_retrieval_chain: 先调 retriever 检索 (纯向量 top3)
    # 把检索结果作为 context 连同 input 传给 qa_chain
    # 由 qa_chain 把 context 塞进 prompt 的 {context} 调LLM生成answer
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

def ask(chain, question):
    """invoke 调用，打印 answer 和 context 里每个 chunk 的页码"""
    result = chain.invoke({"input": question})
    print("=" * 43)
    print("回答: ",result["answer"])
    print("=" * 43)
    print("出处: ")
    for i, doc in enumerate(result["context"]):
        page = doc.metadata.get("page", "未知")
        page_display = page + 1 if isinstance(page, int) else page
        print(f"[{i+1}] 第 {page_display} 页: {doc.page_content[:80]}...]")


if __name__ == "__main__":
    docs = load_pdf(r"D:/Lesley/LLMs架构论文/论文预处理.pdf")
    splits = split_docs(docs)
    vs = build_vectorstore(splits)
    chain = build_chain(vs)
    ask(chain, "SORT算法是什么")