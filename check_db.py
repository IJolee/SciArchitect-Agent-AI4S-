import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "brain_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="science_collection",
    persist_directory=DB_PATH, 
    embedding_function=embeddings
)

# 获取所有元数据
all_metadatas = vectorstore.get()['metadatas']
sources = set([os.path.basename(m.get('source', 'unknown')) for m in all_metadatas])

print("\n📊 当前向量库物理存量清单：")
if not sources:
    print("❌ 警告：数据库是空的！")
else:
    for s in sources:
        print(f"  - {s}")
print(f"\n总计切片数: {len(all_metadatas)}")