import os
import json
import time
import hashlib
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. 核心配置与性能套件 ===
load_dotenv(find_dotenv(), override=True)
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"
DB_PATH = "brain_db"
CACHE_FILE = "assistant_cache.json"

if not API_KEY:
    print("❌ 致命错误：未检测到 API Key。")
    exit()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === 2. 高效检索与指纹去重引擎 ===
class AcademicBrain:
    def __init__(self):
        print("🧠 正在同步科研大脑...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"queries": {}, "retrievals": {}}

    def _save_cache(self):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_hash(self, text):
        # 归一化 hash：去除空白与噪声，取前 500 字生成指纹
        normalized = "".join(text.split()).lower()[:500]
        return hashlib.sha1(normalized.encode()).hexdigest()

    def smart_retrieve(self, query):
        # 第一轮：直接检索
        results = self.db.similarity_search(query, k=5)
        
        # 触发条件：如果首轮结果不足或需要多维视角，执行条件扩容
        if len(results) < 3:
            print("🔍 原始检索深度不足，正在触发 JSON 改写逻辑...")
            expanded_queries = self._expand_query(query)
            for q in expanded_queries:
                results.extend(self.db.similarity_search(q, k=2))

        # 去重逻辑：基于 Metadata + 内容 Hash
        unique_docs = []
        seen_fingerprints = set()
        source_counts = {}

        for doc in results:
            fingerprint = self._get_hash(doc.page_content)
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)

            # 覆盖约束：同一页最多 1 条，同一文件最多 3 条
            source_page_key = f"{source}_{page}"
            if (fingerprint not in seen_fingerprints and 
                source_counts.get(source, 0) < 3 and 
                source_page_key not in seen_fingerprints):
                
                unique_docs.append(doc)
                seen_fingerprints.add(fingerprint)
                seen_fingerprints.add(source_page_key)
                source_counts[source] = source_counts.get(source, 0) + 1

        return unique_docs[:8] # 最终返回 8 条高质量证据卡片

    def _expand_query(self, query):
        if query in self.cache["queries"]:
            return self.cache["queries"][query]
        
        prompt = f"针对学术问题 '{query}'，请给出3个互补的英文学术关键词（包含缩写或变量名），以 JSON 数组格式输出。"
        resp = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        try:
            expanded = json.loads(resp.choices[0].message.content).get("keywords", [])
            self.cache["queries"][query] = expanded
            self._save_cache()
            return expanded
        except:
            return []

# === 3. 对齐闸门与渲染引擎 ===
class ResearchAgent:
    def __init__(self, brain):
        self.brain = brain

    def chat(self, user_query):
        docs = self.brain.smart_retrieve(user_input)
        if not docs:
            return "❌ 库中未发现相关文献，建议补充论文后再试。", []

        # 构建证据卡片
        evidence_cards = ""
        for i, d in enumerate(docs):
            evidence_cards += f"--- [证据 {i+1}] ---\n"
            evidence_cards += f"SOURCE: {d.metadata.get('source')} | PAGE: {d.metadata.get('page',0)+1}\n"
            evidence_cards += f"CONTENT: {d.page_content.strip()}\n\n"

        # 阶段 A：Claim->Evidence 硬对齐 (JSON 内部旁路)
        alignment_prompt = f"""
        你是一位计算机视觉与大模型领域的资深专家。
        基于以下【证据卡片】，首先判断证据是否足以回答问题。
        如果足以回答，请先生成一个结构化的【对齐表】，然后基于对齐表撰写一份深度的科研分析报告。

        【证据卡片】：
        {evidence_cards}

        【输出要求】：
        你必须以 JSON 格式输出，包含以下字段：
        1. "sufficiency": true/false (证据是否充分)
        2. "alignment": [{"claim": "结论", "evidence_id": [数字]}] (结论与证据的硬绑定)
        3. "detailed_report": "按照提问者的风格，撰写有深度、多维度的分析，字数不限。"
        """

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "system", "content": "你只输出 JSON。"},
                      {"role": "user", "content": alignment_prompt}],
            response_format={"type": "json_object"}
        )
        
        res_data = json.loads(response.choices[0].message.content)
        
        if not res_data.get("sufficiency"):
            return f"⚠️ 证据不足。缺失信息：{res_data.get('why', '核心参数未提及')}", []

        return res_data["detailed_report"], [f"{d.metadata.get('source')} (P{d.metadata.get('page',0)+1})" for d in docs]

# === 4. 运行入口 ===
if __name__ == "__main__":
    brain = AcademicBrain()
    agent = ResearchAgent(brain)
    
    print("\n🚀 AI4S 科研助手 V2 已上线。输出风格已同步。")
    while True:
        user_input = input("\n🙋 你: ").strip()
        if user_input.lower() in ['q', 'exit']: break
        
        print("⏳ 正在执行硬核对齐检索...")
        start_time = time.time()
        report, sources = agent.chat(user_input)
        
        print("\n" + "="*60)
        print(f"🤖 AI 深度研报:\n\n{report}")
        print("\n" + "="*60)
        print("📚 引用追踪:")
        for s in set(sources):
            print(f"  - {s}")
        print(f"\n(分析耗时: {time.time() - start_time:.2f}s)")