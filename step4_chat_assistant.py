import os
import json
import time
import hashlib
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =========================
# 1. 核心配置 (Academic Aesthetic)
# =========================
# 强行寻找 .env 文件并覆盖环境变量
load_dotenv(find_dotenv(), override=True)

# 【防弹级密钥获取】
API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not API_KEY:
    raise ValueError("🚨 致命错误：未能从 .env 文件中读取到 SILICONFLOW_API_KEY，请确保项目目录下存在 .env 文件并填入了密钥！")

BASE_URL = "https://api.siliconflow.cn/v1"
DB_PATH = "brain_db"
DOCSTORE_PATH = "./docstore_data" 
CACHE_FILE = "assistant_cache.json"
# 动态配额与阈值
MAX_CARD_CHARS = 1300 
FIRST_K, EXPAND_K_PER_QUERY, MAX_EXPANDED_QUERIES, MAX_FINAL_DOCS = 6, 2, 3, 8

if not API_KEY: raise SystemError("❌ 致命错误：未检测到 API Key")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =========================
# 2. 增强型工具套件 (SHA1 + JSON Safety)
# =========================
def sha1(text: str) -> str: return hashlib.sha1(text.encode("utf-8")).hexdigest()

def normalize_fingerprint(text: str, take: int = 600) -> str:
    """去噪后的文本指纹，防止重复检索"""
    t = re.sub(r"\s+", "", text).lower()
    return sha1(t[:take])

def safe_load_json(text: str) -> Dict[str, Any]:
    """增强版 JSON 解析，专门处理大模型输出的不规范格式"""
    s = text.strip()
    s = re.sub(r"```json\s*|```\s*", "", s)
    try:
        return json.loads(s)
    except:
        match = re.search(r"(\{.*\})", s, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
    raise ValueError("DeepSeek 返回格式非标准 JSON，请重试或检查 Prompt 约束")

def clip_text(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"

# =========================
# 3. AcademicBrain：持久化父子回溯引擎 + 文件名雷达
# =========================
class AcademicBrain:
    def __init__(self):
        print("🧠 正在同步科研大脑 (Parent-Doc Persistent)...")
        
        # 1. 视神经 (Embedding)
        self.embeddings = OpenAIEmbeddings(
            api_key=API_KEY,          # 直接用顶部的安全变量
            base_url=BASE_URL,        # 直接用顶部的安全变量
            model="BAAI/bge-m3",
            chunk_size=64             # 👈 绝对不能漏掉这个防爆破参数！
        )
        
        # 2. 挂载向量数据库
        self.vectorstore = Chroma(
            collection_name="science_collection",
            persist_directory=DB_PATH, 
            embedding_function=self.embeddings
        )
        
        # 3. 挂载文档存储
        if not os.path.exists(DOCSTORE_PATH): 
            os.makedirs(DOCSTORE_PATH)
        fs = LocalFileStore(DOCSTORE_PATH)
        
        self.store = EncoderBackedStore(
            store=fs,
            key_encoder=lambda k: hashlib.sha1(k.encode()).hexdigest(),
            value_serializer=lambda v: json.dumps({"page_content": v.page_content, "metadata": v.metadata}).encode(),
            value_deserializer=lambda v: Document(**json.loads(v.decode()))
        )
        
        # 4. 初始化父子文档检索器
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200),
        )
        
        # 5. 加载本地缓存
        self.cache = self._load_cache() # 👈 恢复你的缓存加载

    def _load_cache(self) -> Dict[str, Any]:
        """恢复：本地查询缓存加载"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f: return json.load(f)
            except: pass
        return {"queries": {}, "retrievals": {}}

    def _save_cache(self):
        """恢复：本地查询缓存保存"""
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _expand_query(self, query: str) -> List[str]:
        """恢复：多路改写逻辑，提高召回率"""
        if query in self.cache["queries"]: return self.cache["queries"][query]
        prompt = f"针对学术提问：{query}\n输出JSON，包含'keywords'数组（学术缩写/机制名）。"
        resp = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "system", "content": "你只输出JSON。"}, {"role": "user", "content": prompt}]
        )
        try:
            kws = safe_load_json(resp.choices[0].message.content).get("keywords", [])[:MAX_EXPANDED_QUERIES]
        except: kws = []
        self.cache["queries"][query] = kws
        self._save_cache()
        return kws

    def smart_retrieve(self, query: str) -> List[Any]:
        """
        全功能检索引擎：融合广域召回、父文档回溯、雷达提权与弹性多样性保护
        """
        # 1. 广域初始召回：从向量库拿 30 个候选，为后续的“平衡重排”留足空间
        # 注意：similarity_search_with_relevance_scores 返回 (doc, score)
        raw_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=30)
        
        # 2. 准备候选池并回溯父文档 (Parent-Doc 逻辑)
        candidate_pool = []
        for doc, score in raw_results:
            # 过滤掉分数极低（完全不相关）的内容
            if score < 0.1: continue 
            
            # 尝试回溯父文档以获取更完整的上下文
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                parent_doc = self.store.mget([doc_id])[0]
                if parent_doc: 
                    doc = parent_doc
            
            candidate_pool.append({"doc": doc, "score": score})

        # 如果初始召回太少，触发你之前的扩展查询逻辑
        if len(candidate_pool) < 5:
            print("🔍 原始信号较弱，尝试扩展查询以增强召回...")
            expanded_queries = self._expand_query(query)
            for eq in expanded_queries:
                eq_results = self.vectorstore.similarity_search_with_relevance_scores(eq, k=10)
                for d, s in eq_results:
                    # 同样执行父文档回溯
                    d_id = d.metadata.get("doc_id")
                    if d_id:
                        p_doc = self.store.mget([d_id])[0]
                        if p_doc: d = p_doc
                    candidate_pool.append({"doc": d, "score": s})

        # 3. ⚖️ 弹性惩罚与雷达重排 (Diversity & Radar Logic)
        unique_docs = []      # 最终录用的文档列表
        seen_fp = set()       # 内容指纹去重集
        source_counts = {}    # 记录各来源已入选次数

        # 识别用户是否在问某篇具体的论文
        radar_keywords = ["deepseek", "2602", "transformer", "loftr", "lara", "论文7", "manual"]
        lower_query = query.lower()

        # 对候选池进行二次打分
        for item in candidate_pool:
            doc = item['doc']
            source = os.path.basename(doc.metadata.get("source", "unknown")).lower()
            
            # --- 弹性惩罚：每多出一个切片，该来源后续切片的权重就打 0.5 折 ---
            count = source_counts.get(source, 0)
            penalty = 0.5 ** count
            
            # --- 雷达提权：如果是用户点名的核心论文，给予权重补偿 ---
            boost = 1.0
            if any(k in source for k in radar_keywords if k in lower_query):
                boost = 1.5
            
            # 特殊处理：如果不是点名要看 manual，给 manual 额外的负向修正
            if "manual" in source and "manual" not in lower_query:
                boost *= 0.2  # 压制非必要的说明书内容

            item['final_score'] = item['score'] * penalty * boost

        # 4. 按照调整后的 final_score 重新从高到低排序
        candidate_pool.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        # 5. 🚀 最终收割与内容去重
        for item in candidate_pool:
            doc = item['doc']
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            
            # 计算内容指纹，防止相似段落刷屏
            fp = normalize_fingerprint(doc.page_content)
            if fp in seen_fp: 
                continue
            
            # 录用证据
            unique_docs.append(doc)
            seen_fp.add(fp)
            
            # 更新该来源的计数，供下一轮计算（理论上已排好序，此处计数主要用于日志或调试）
            source_counts[source] = source_counts.get(source, 0) + 1
            
            # 凑够 8 条最优质且多样化的证据就停止
            if len(unique_docs) >= 8: # 8 即 MAX_FINAL_DOCS
                break
            
        return unique_docs


# =========================
# 4. ResearchAgent：博导级弹性溯源 (解除幻觉束缚)
# =========================
class ResearchAgent:
    def __init__(self, brain: AcademicBrain): 
        self.brain = brain

    def chat(self, user_query: str) -> Tuple[str, List[str]]:
        docs = self.brain.smart_retrieve(user_query)
        if not docs: 
            return "❌ 库中未发现证据", []
        
        cards = []
        for i, d in enumerate(docs, 1):
            source = os.path.basename(d.metadata.get("source","unknown"))
            page = int(d.metadata.get("page",0))+1
            content = clip_text(d.page_content, MAX_CARD_CHARS)
            cards.append(f"--- [证据 {i}] ---\n【本段来源：{source} | 第 {page} 页】\n详情：{content}\n")
        
        # 🌟 替换为顶级架构师 Prompt，【全局 AI4S 泛化版】防幻觉与跨模态阻断协议
        prompt = f"""你不是“论文总结器”，而是【科研问答官与顶级架构师】。
默认前提：用户已经读过论文，希望你解释**机制因果、数学约束与逻辑差异**。

==================== 用户提问 ====================
{user_query}

==================== 可用证据 ====================
（注意：本地检索器可能会召回完全无关的说明书。请仔细甄别！）
{chr(10).join(cards)}

==================== 🚨 答辩执行准则（严苛防幻觉与跨域阻断协议 - 通用版）====================
1. 【变量与领域前置审查】：如果用户强行组合不同领域的概念（例如：将A领域的物理约束/算法直接套用于B领域），你【必须】首先在脑内审查该设定的合法性。
   - 必须审视输入输出空间是否对齐、张量维度是否匹配、假设前提是否冲突。
   - 如果用户没有明确定义跨域映射关系，你绝对不可顺着错误前提脑补无关的领域特定概念。必须指出“变量未定义或模态错位”。
2. 【禁止伪数学与乱引数据集】：
   - 当用户问及“数学上是否根除/保证”时，严禁捏造虚假阈值或不存在的定理。
   - 严禁强行引用与当前讨论领域不匹配的数据集作为证明。
   - 必须从“必要条件、充分条件、反例、优化目标的非凸性/不完备性”等严格数学逻辑层面给出论证。
3. 【本地证据偏题处理】：
   - 若证据无关，请在各项“证据锚定”处声明“[本地证据偏题]”。动用自身知识推演时，宁可得出“理论上无法保证”或“跨域设定存在漏洞”的结论，也【绝对禁止】堆砌华丽的学术词汇造假。

==================== 输出任务 ====================
请严格遵循以下结构与格式输出（必须输出合法的 JSON，值为包含以下Markdown结构的文本）：

------------------------------------------------
A) 一句话核心结论（≤60字）
------------------------------------------------
- 直接回答因果答案或数学可行性。如果设定本身变量缺失/跨域冲突，直接指出“设定存在模态错位，无法给出理论保证”。

------------------------------------------------
B) 机制级拆解（3～5 点，重点）
------------------------------------------------
每一点必须使用以下固定格式：

**【机制要点标题】**（8～14字）
- **因果链**：因为 → 所以 → 从而（必须点名具体机制/算子/约束，如果跨域移植不成立，请说明物理/数学上为什么不成立）
- **解决的歧义/失败模式**：明确说原提问忽视了什么致命问题（如：维度灾难、分布偏移、目标函数不一致等）。
- **证据锚定或推演逻辑**：如果证据相关，格式为 `[证据x｜文件名P页] "原句"`；如果证据偏题，给出**严谨的数学反例**或**领域自洽的最小验证实验(MVE)**。

------------------------------------------------
C) 与传统方法的“逻辑差异”对比（必须）
------------------------------------------------
- 对比对象：传统方法 vs 用户提及的新跨域方法
- 对比维度：**先验假设的差异 / 约束施加位置 / 核心失败模式**

------------------------------------------------
D) 结合真实应用/工程场景（必须）
------------------------------------------------
用 2～4 行回答：
- 在什么极端/长尾场景下这个方案会彻底失效？
- 引入这种设计的真实物理算力/内存空间代价是什么？

------------------------------------------------
E) 深度追问（2～3 条，必须）
------------------------------------------------
你必须在结尾主动向用户提出追问，引向更深一层的工程死穴（如张量如何对齐）或理论边界。

------------------------------------------------
输出格式：必须是严格合法的 JSON（值全为中文）
{{
  "analysis_focus": "一句话核心结论",
  "detailed_report": "包含 A/B/C/D/E 五个模块的 Markdown 正文"
}}
"""

        resp = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "你必须只输出合法的 JSON，内容要极其硬核，绝不敷衍。注意使用 \\\" 转义双引号。"}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, 
            response_format={"type": "json_object"}
        )
        
        try:
            obj = safe_load_json(resp.choices[0].message.content)
            report = obj.get("detailed_report", "分析完成")
            if isinstance(report, dict):
                report = json.dumps(report, ensure_ascii=False, indent=2)
            sources = [f"{os.path.basename(d.metadata.get('source'))} (P{d.metadata.get('page',0)+1})" for d in docs]
            return str(report), sorted(set(sources))
        except Exception as e: 
            return f"❌ 对齐渲染失败: {e}", []

# =========================
# 5. CLI 运行入口
# =========================
if __name__ == "__main__":
    brain = AcademicBrain()
    agent = ResearchAgent(brain)
    print("\n🚀 AI4S 科研助手 V2.3 (Ultra-Robust + Architect) 就绪。")
    while True:
        user_input = input("\n🙋 你: ").strip()
        if user_input.lower() in ["q", "exit"]: break
        if not user_input: continue
        
        print("⏳ 正在博导级对齐分析...")
        t0 = time.time()
        report, sources = agent.chat(user_input)
        
        print("\n" + "="*70 + "\n🤖 研报输出：\n\n" + report + "\n" + "="*70)
        
        if sources:
            print("\n📚 本次触发雷达的底层切片：")
            for s in sources: print(f"  - {s}")
        print(f"\n(耗时: {time.time() - t0:.2f}s)")
