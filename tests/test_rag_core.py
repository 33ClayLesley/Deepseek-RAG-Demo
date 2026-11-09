import pytest
import hashlib
from unittest.mock import MagicMock
from rag_core import RAGEngine
from config import settings

@pytest.fixture
def mock_engine(mocker):
    """Fixture：创建一个 mock 掉 Embedding 和 LLM 初始化的引擎实例"""
    # 拦截 HuggingFaceEmbeddings 和 ChatOpenAI，防止测试时下载模型或调用 API
    mocker.patch("rag_core.HuggingFaceEmbeddings")
    mocker.patch("rag_core.ChatOpenAI")
    
    # 实例化引擎
    engine = RAGEngine()
    return engine

def test_file_hash_logic(mock_engine):
    """测试 1：Hash 去重逻辑是否正确"""
    file_bytes = b"fake pdf content"
    
    # 手动计算预期的 Hash (包含模型名称 + 文件内容)
    expected_hash = hashlib.md5(settings.embedding_model.encode('utf-8') + file_bytes).hexdigest()
    
    # 调用引擎内部方法计算
    actual_hash = mock_engine._get_file_hash(file_bytes)
    
    assert actual_hash == expected_hash

def test_process_pdf_cache_hit(mock_engine, mocker):
    """测试 2：缓存命中时，不应该去解析 PDF"""
    # 伪造前端上传的文件对象
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "test.pdf"
    mock_uploaded_file.getvalue.return_value = b"fake content"

    # Mock Chroma 实例，模拟已有数据（count > 0）
    mock_chroma_instance = MagicMock()
    mock_chroma_instance._collection.count.return_value = 10 # 模拟已经有 10 条数据了
    mocker.patch("rag_core.Chroma", return_value=mock_chroma_instance)

    # 监视 PyPDFLoader 是否被调用
    mock_pdf_loader = mocker.patch("rag_core.PyPDFLoader")

    # 执行处理
    vectorstore = mock_engine.process_pdf(mock_uploaded_file)

    # 断言：直接返回了实例，且根本没有去解析 PDF
    assert vectorstore == mock_chroma_instance
    mock_pdf_loader.assert_not_called()

def test_process_pdf_cache_miss(mock_engine, mocker):
    """测试 3：没有缓存时，走完整解析流程"""
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "new_doc.pdf"
    mock_uploaded_file.getvalue.return_value = b"new content"

    # Mock Chroma 实例，模拟没有数据（count == 0）
    mock_chroma_instance = MagicMock()
    mock_chroma_instance._collection.count.return_value = 0
    mocker.patch("rag_core.Chroma", return_value=mock_chroma_instance)

    # Mock PyPDFLoader 和 Splitter
    mock_pdf_loader = mocker.patch("rag_core.PyPDFLoader")
    # 显式指定 page_content 为字符串
    mock_pdf_loader.return_value.load.return_value = [MagicMock(page_content="fake document text", metadata={})]
    
    # 直接替换实例的切分方法
    mock_engine.text_splitter.split_documents = MagicMock(return_value=["chunk1", "chunk2"])

    # 执行处理
    vectorstore = mock_engine.process_pdf(mock_uploaded_file)

    # 断言：流程完整走完，add_documents 被调用
    assert vectorstore == mock_chroma_instance
    mock_pdf_loader.assert_called_once()
    mock_chroma_instance.add_documents.assert_called_once_with(["chunk1", "chunk2"])

def test_get_qa_chain_requires_vectorstore(mock_engine):
    """测试 4：如果没有向量库就直接调用问答链，应该抛出异常"""
    mock_engine.vectorstore = None
    with pytest.raises(ValueError, match="Vectorstore 尚未初始化"):
        mock_engine.get_qa_chain()

def test_empty_file_error(mock_engine, mocker):
    """测试 5：空文件异常处理"""
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.getvalue.return_value = b""

    mock_chroma_instance = MagicMock()
    mock_chroma_instance._collection.count.return_value = 0
    mocker.patch("rag_core.Chroma", return_value=mock_chroma_instance)

    # 让 PyPDFLoader 抛出异常
    mocker.patch("rag_core.PyPDFLoader", side_effect=Exception("PDF 解析失败"))

    with pytest.raises(Exception, match="PDF 解析失败"):
        mock_engine.process_pdf(mock_uploaded_file)