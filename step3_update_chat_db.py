import os
import shutil
import json
import hashlib
import time
from dotenv import load_dotenv, find_dotenv  # 引入了 find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# === 升维配置：持久化科研索引 ===
# 强行寻找 .env 文件并覆盖环境变量
load_dotenv(find_dotenv(), override=True)
DB_PATH = "./brain_db"
DOCSTORE_PATH = "./docstore_data"
LIBRARY_FILE = "library.json"

# 【防弹级密钥获取】防止任何作用域变量丢失
API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not API_KEY:
    # 彻底干掉硬编码，读不到密钥直接抛出红色警报并停止运行
    raise ValueError("🚨 致命错误：未能从 .env 文件中读取到 SILICONFLOW_API_KEY，请确保项目目录下存在 .env 文件并填入了密钥！")

def build_science_brain():
    print("=== 🧠 AI4S-Agent: Step 3 - 持久化多语言科研知识库构建 ===")
    
    if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
    if os.path.exists(DOCSTORE_PATH): shutil.rmtree(DOCSTORE_PATH)
    os.makedirs(DOCSTORE_PATH)

    library_data = {}
    if os.path.exists(LIBRARY_FILE):
        with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
            library_data = json.load(f)

    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if not pdf_files:
        print("❌ 错误：未发现 PDF 论文。")
        return

    all_docs = []
    for pdf in pdf_files:
        print(f"  -> 正在解析并注入元数据: {pdf} ...")
        try:
            loader = PyPDFLoader(pdf)
            pages = loader.load()
            category = library_data.get(pdf, {}).get("category", "General")
            for page in pages:
                page.metadata["category"] = category
                page.metadata["source"] = pdf
            all_docs.extend(pages)
        except Exception as e:
            print(f"     ⚠️ 解析 {pdf} 时出现警告: {e}")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    # 1. 强制声明 chunk_size，确保单次 HTTP Payload 在硅基流动承受范围内
    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,  # 直接使用顶部安全获取到的变量
        base_url="https://api.siliconflow.cn/v1", # 或者你新 API 的基地址
        model="BAAI/bge-m3", # 必须是这个向量模型，不要写 DeepSeek
        chunk_size=64
    )
    
    # ... (你原本代码下方关于 Chroma、Retriever 的挂载逻辑请保持原样，无需改动) ...    
    vectorstore = Chroma(
        collection_name="science_collection", 
        embedding_function=embeddings, 
        persist_directory=DB_PATH
    )
    
    fs = LocalFileStore(DOCSTORE_PATH)
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda k: hashlib.sha1(k.encode()).hexdigest(),
        value_serializer=lambda v: json.dumps({"page_content": v.page_content, "metadata": v.metadata}).encode(),
        value_deserializer=lambda v: Document(**json.loads(v.decode()))
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print(f"\n🚀 准备就绪，共扫描到 {len(all_docs)} 页 PDF，正在执行多级索引挂载...")
    print("⏳ 为防止触发线上 API 封禁，采用平滑分批上传策略 (Batching)...")
    
    # 2. 应用层防爆破：将所有页面切分为每 10 页一个批次缓慢推送
    batch_size = 10 
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i+batch_size]
        print(f"  -> 正在映射张量... 进度: 第 {i+1} 页 至 第 {min(i+batch_size, len(all_docs))} 页")
        retriever.add_documents(batch, ids=None)
        time.sleep(0.5) # 加上呼吸延时，躲避高频检测雷达
        
    print(f"\n✅ 超级知识库构建完毕！BGE-M3 多语言特征已持久化至 {DB_PATH}")

if __name__ == "__main__":
    build_science_brain()