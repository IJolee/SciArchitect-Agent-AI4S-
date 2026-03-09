import os
import json
import re
import time
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader

# =========================================================
# Step2 V23：论文“细分精读 + 全要点收敛 + 追问闭环”系统（中文输出）
# 产物：Markdown 研报（末尾带“可追问问题”）
# 交互：生成后可直接进入追问（本论文证据 / Step4 向量库）
# =========================================================

load_dotenv()
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"
LIBRARY_FILE = "library.json"

# ---------- 采样配置：覆盖方法/实验/训练设置/部署通信/附录 ----------
WEIGHTS: Dict[str, Tuple[int, int]] = {
    # 方法主线
    "method": (12, 24),
    "algorithm": (12, 24),
    "architecture": (12, 24),
    "loss": (12, 24),
    "attention": (10, 20),
    "moe": (10, 20),
    "routing": (10, 20),

    # 实验与评估
    "experiment": (14, 28),
    "evaluation": (14, 28),
    "result": (10, 20),
    "metric": (12, 24),
    "baseline": (12, 24),
    "dataset": (12, 24),
    "ablation": (22, 36),
    "table": (8, 16),
    "figure": (8, 16),

    # 训练与超参
    "training": (14, 28),
    "setup": (14, 28),
    "hyperparameter": (14, 28),
    "optimizer": (12, 24),
    "schedule": (12, 24),
    "learning rate": (12, 24),
    "batch": (10, 20),
    "epoch": (10, 20),

    # 工程实现/并行/精度/通信
    "implementation": (12, 24),
    "pipeline": (12, 24),
    "parallel": (12, 24),
    "all-to-all": (14, 28),
    "nvlink": (12, 24),
    "infiniband": (12, 24),
    "fp8": (18, 32),
    "quant": (14, 28),

    # 局限/失败/附录
    "limitation": (22, 36),
    "failure": (22, 36),
    "appendix": (16, 28),
}

TOP_K_ANCHORS = 14
PAD_TOP_N = 4
MAX_PAGES = 22

MODEL_NAME = "deepseek-ai/DeepSeek-V3"
TEMPERATURE = 0.1

# 追问模式参数
FOLLOWUP_MAX_TURNS = 50
FOLLOWUP_MAX_EVIDENCE_CHARS = 14000  # 避免上下文爆炸


# =========================
# 1) 工具函数
# =========================
def _score_page(text: str) -> int:
    t = text.lower()
    score = 0
    for kw, (val, cap) in WEIGHTS.items():
        score += min(t.count(kw) * val, cap)
    return score


def _best_page_by_keywords(docs, kws: List[str]) -> Optional[int]:
    best_i, best_sc = None, -1
    for i, d in enumerate(docs):
        t = d.page_content.lower()
        hit = sum(1 for k in kws if k in t)
        sc = _score_page(d.page_content) + hit * 10
        if sc > best_sc:
            best_sc = sc
            best_i = i
    return best_i


def _pick_indices(docs) -> List[int]:
    scored: List[Tuple[int, int]] = []
    for i, d in enumerate(docs):
        sc = _score_page(d.page_content)
        if i < 2:
            sc += 60  # 强保前2页主线
        scored.append((i, sc))

    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    anchors = [i for i, _ in scored_sorted[:TOP_K_ANCHORS]]

    final = set(anchors)

    # padding（只对最高分前 PAD_TOP_N）
    for i, _ in scored_sorted[:PAD_TOP_N]:
        if i > 0:
            final.add(i - 1)
        if i < len(docs) - 1:
            final.add(i + 1)

    # 强制覆盖：方法页 + 实验页（避免“只抓到附录/只抓到背景”）
    method_idx = _best_page_by_keywords(
        docs, ["method", "algorithm", "architecture", "loss", "routing", "attention", "moe"]
    )
    exper_idx = _best_page_by_keywords(
        docs, ["experiment", "evaluation", "result", "table", "metric", "baseline", "dataset", "ablation"]
    )
    if method_idx is not None:
        final.add(method_idx)
    if exper_idx is not None:
        final.add(exper_idx)

    # 得分优先裁剪到 MAX_PAGES（拼接时再按页码排序）
    score_dict = dict(scored)
    final_scored = [(i, score_dict.get(i, 0)) for i in final]
    keep = [i for i, _ in sorted(final_scored, key=lambda x: x[1], reverse=True)[:MAX_PAGES]]

    return sorted(keep)


def _build_context(docs, indices: List[int]) -> str:
    parts = []
    for idx in indices:
        p = idx + 1
        parts.append(f"\n--- [PAGE {p}] ---\n{docs[idx].page_content}\n")
    return "".join(parts)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _parse_json_strict(s: str) -> Dict[str, Any]:
    s = s.strip()
    # 暴力清理大模型可能带有的 Markdown 格式块
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    s = s.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError as e:
        # 兜底：如果还是失败，尝试只截取大括号里面的内容
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                obj = json.loads(s[l:r + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception as inner_e:
                raise ValueError(f"❌ 模型输出的 JSON 语法彻底损坏：\n{s[l:r+1]}\n错误信息: {inner_e}")
        raise ValueError(f"❌ 模型输出不是可解析的 JSON。错误: {e}")


# =========================
# 2) Schema 校验：保证“可答辩结构”稳定产出
# =========================
# =========================
# 2) Schema 校验：严格匹配你的“重型全解构”Prompt
# =========================
def _validate_schema(data: Dict[str, Any]) -> None:
    # 1. 顶层字段
    need = ["readme_tagline", "summary", "table_map", "audit_cards", "repro_card", "interactive_battle_guide", "cited_pages"]
    for k in need:
        if k not in data:
            raise KeyError(f"❌ JSON 缺顶层字段：{k}")

    # 2. summary
    for k in ["problem", "method", "result", "evidence_pages"]:
        if k not in data["summary"]:
            raise KeyError(f"❌ summary 缺字段：{k}")

    # 3. table_map
    tm = data["table_map"]
    for k in ["pipeline_table", "mechanism_table", "reading_order"]:
        if k not in tm:
            raise KeyError(f"❌ table_map 缺字段：{k}")
    if not isinstance(tm["pipeline_table"], list) or len(tm["pipeline_table"]) == 0:
        raise ValueError("❌ pipeline_table 必须非空")
    if not isinstance(tm["mechanism_table"], list) or len(tm["mechanism_table"]) == 0:
        raise ValueError("❌ mechanism_table 必须非空")

    # 4. audit_cards
    if not isinstance(data["audit_cards"], list) or len(data["audit_cards"]) == 0:
        raise ValueError("❌ audit_cards 必须是非空列表。")
    for i, c in enumerate(data["audit_cards"]):
        for kk in ["title", "pain_point", "evidence", "inference", "simple_translation"]:
            if kk not in c:
                raise ValueError(f"❌ audit_cards[{i}] 缺字段：{kk}")
        e = c["evidence"]
        inf = c["inference"]
        for kk in ["page", "quote", "interpretation"]:
            if kk not in e:
                raise ValueError(f"❌ audit_cards[{i}].evidence 缺字段：{kk}")
        for kk in ["assumptions", "impact_range", "quick_test"]:
            if kk not in inf:
                raise ValueError(f"❌ audit_cards[{i}].inference 缺字段：{kk}")

    # 5. repro_card
    rc = data["repro_card"]
    for kk in ["known_from_paper", "mentor_fillins", "single_gpu_plan", "fallback"]:
        if kk not in rc:
            raise ValueError(f"❌ repro_card 缺字段：{kk}")

    # 6. 交互对线指南
    if "suggested_queries" not in data["interactive_battle_guide"]:
        raise KeyError("❌ interactive_battle_guide 缺字段：suggested_queries")


# =========================
# 3) Markdown 渲染：高信息密度卡片式排版
# =========================
def _render_md_cn(pdf_name: str, data: Dict[str, Any], followups: List[Dict[str, Any]]) -> str:
    md: List[str] = []
    md.append(f"# 📄 论文全要点精读研报：{pdf_name}\n\n")
    md.append(f"> **💡 核心亮点**: {data.get('readme_tagline', '')}\n\n")

    # ================= 0) 新增：最前方的概括总结 =================
    s = data["summary"]
    md.append("## 🌟 0. 论文核心概括\n")
    md.append(f"**【背景与痛点】** {s.get('problem','')}\n\n")
    md.append(f"**【最终战果】** {s.get('result','')}\n\n")

    # ================= 1) 逻辑链条 =================
    md.append("## 🌊 1. 核心逻辑执行链条\n")
    md.append(f"{s.get('method','')}\n\n")
    md.append(f"- **证据页码**：{s.get('evidence_pages', [])}\n\n")

    # ================= 2) 废弃表格，改用卡片列表 =================
    tm = data["table_map"]
    md.append("## 🗺️ 2. 全流程解构地图\n\n")
    
    # --- 2.1 Pipeline 文字卡片 ---
    md.append("### ⚙️ 2.1 Pipeline 底层流向拆解\n")
    for i, row in enumerate(tm["pipeline_table"], 1):
        md.append(f"**🔹 阶段 {i}：{row.get('stage','')}**\n")
        md.append(f"- **📥 输入**：{row.get('input','')}\n")
        md.append(f"- **🧠 核心齿轮**：{row.get('core_op','')}\n")
        md.append(f"- **📤 输出**：{row.get('output','')}\n")
        md.append(f"- **🩸 隐形代价**：{row.get('cost','')}\n")
        md.append(f"- **📄 证据**：`{row.get('evidence','')}`\n\n")
    
    # --- 2.2 Mechanism 文字卡片 ---
    md.append("### 🔬 2.2 Mechanism 痛点突破与机制\n")
    for i, row in enumerate(tm["mechanism_table"], 1):
        md.append(f"**🔹 机制 {i}：{row.get('mechanism','')}**\n")
        md.append(f"- **💡 原理解析**：{row.get('what_it_does','')}\n")
        md.append(f"- **🎯 解决痛点**：{row.get('why_needed','')}\n")
        md.append(f"- **⚠️ 物理上限/失效场景**：{row.get('hidden_cost','')}\n")
        md.append(f"- **🍎 生活化类比**：{row.get('how_to_verify','')}\n")
        md.append(f"- **📄 证据**：`{row.get('evidence','')}`\n\n")
    
    md.append("### 📖 2.3 推荐阅读顺序\n")
    for x in tm.get("reading_order", []):
        md.append(f"- {x}\n")
    md.append("\n")

    # ================= 3) 审计官卡片 =================
    md.append("## 🔍 3. 首席审计员卡片（证据 vs 工程推断）\n")
    for i, card in enumerate(data["audit_cards"], 1):
        md.append(f"### [{i}] {card.get('title','')}\n")
        md.append(f"- **核心矛盾**：{card.get('pain_point','')}\n")
        e = card["evidence"]
        inf = card["inference"]
        md.append(f"- **原文证据**：[PAGE {e.get('page')}] `{e.get('quote','')}`\n")
        md.append(f"  - 通俗解构：{e.get('interpretation','')}\n")
        md.append(f"- **工程推断**：\n")
        md.append(f"  - 假设前提：{inf.get('assumptions','')}\n")
        md.append(f"  - 影响区间：{inf.get('impact_range','')}\n")
        md.append(f"  - 1小时验证：{inf.get('quick_test','')}\n")
        md.append(f"- **范式转移**：{card.get('simple_translation','')}\n\n")

    # ================= 4) 复现卡 =================
    md.append("## 🛠️ 4. 复现与边界预判\n")
    rc = data["repro_card"]
    md.append("### 4.1 原文明确的设置\n")
    for x in rc.get("known_from_paper", []):
        md.append(f"- {x}\n")
    md.append("\n### 4.2 审计官质疑（隐藏副作用）\n")
    for x in rc.get("mentor_fillins", []):
        md.append(f"- {x}\n")
    md.append("\n### 4.3 最强证据\n")
    md.append(f"{rc.get('single_gpu_plan','')}\n\n")
    md.append("### 4.4 边界条件与极端失效场景\n")
    md.append(f"{rc.get('fallback','')}\n\n")

    md.append(f"## 📌 5. 本次引用页码\n- {data.get('cited_pages', [])}\n\n")

    # ================= 5) 动态追问 =================
    md.append("## ❓ 6. 审计台动态追问列表\n")
    battle_guide = data.get("interactive_battle_guide", {}).get("suggested_queries", [])
    for i, q in enumerate(battle_guide, 1):
        md.append(f"{i}. **{q.get('query', '')}**\n")
        md.append(f"   - 审计动机：{q.get('reason', '')}\n")
        md.append(f"   - 关联证据页：P{q.get('cite_page', '未知')}\n")
    md.append("\n")

    return "".join(md)
# =========================
# 4) 生成“可追问问题”（使用大模型原生生成的硬核质疑）
# =========================
def _generate_followups_heuristic(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    废弃旧的正则硬编码提取逻辑。
    直接透传大模型在 JSON 结构中生成的 `interactive_battle_guide` 高质量提问。
    """
    return data.get("interactive_battle_guide", {}).get("suggested_queries", [])
# =========================
# 5) 追问模式 A：只基于本论文上下文（严格证据）
# =========================
def _followup_chat_paper_only(client: OpenAI, pdf_name: str, context_text: str):
    print("\n🧩 已进入追问模式 A（只基于本论文采样页，严格证据，不脑补）")
    print("输入 q 退出。\n")

    # 保护：上下文过长就裁剪
    ctx = context_text
    if len(ctx) > FOLLOWUP_MAX_EVIDENCE_CHARS:
        ctx = ctx[:FOLLOWUP_MAX_EVIDENCE_CHARS] + "\n[...证据过长已裁剪...]"

    system_prompt = f"""
你是一位严谨的科研导师，只能基于给定分页文本回答，不许脑补论文外信息。
回答规则：
1) 结论先行（1-2句）
2) 每条关键断言必须附 [PAGE X] 引文（<=60字）
3) 没证据就说“原文未提供”
分页文本如下：
{ctx}
"""

    turns = 0
    while turns < FOLLOWUP_MAX_TURNS:
        q = input("🙋 你问：").strip()
        if not q:
            continue
        if q.lower() in ["q", "quit", "exit"]:
            break

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"论文：{pdf_name}\n问题：{q}"}
            ],
            temperature=0.1
        )
        print("\n🤖 回答：\n" + resp.choices[0].message.content.strip() + "\n")
        turns += 1


# =========================
# 6) 追问模式 B：接入 Step4（向量库检索追问）
# =========================
def _followup_chat_step4():
    print("\n🧠 尝试接入追问模式 B（Step4 向量库检索）...")

    try:
        # 你现有的 step4 文件里已经定义了 AcademicBrain/ResearchAgent
        # 并且具备“来源硬绑定 + JSON 安全解析”等关键补丁
        # 见：ResearchAgent.chat 与证据卡片构建逻辑
        from step4_chat_assistant import AcademicBrain, ResearchAgent
    except Exception as e:
        print(f"❌ 未能导入 step4_chat_assistant.py：{e}")
        print("请确认 step4_chat_assistant.py 与本文件同目录。")
        return

    brain = AcademicBrain()
    agent = ResearchAgent(brain)

    print("✅ 已进入追问模式 B（向量库检索）。输入 q 退出。\n")
    turns = 0
    while turns < FOLLOWUP_MAX_TURNS:
        q = input("🙋 你问：").strip()
        if not q:
            continue
        if q.lower() in ["q", "quit", "exit"]:
            break
        report, sources = agent.chat(q)
        print("\n🤖 回答：\n" + str(report).strip())
        if sources:
            print("\n📚 引用追踪：")
            for s in sources:
                print(f"  - {s}")
        print()
        turns += 1


# =========================
# 7) 主流程：精读 + 追加问题 + 可选追问
# =========================
# =========================
# 7) 主流程：精读 + 追加问题 + 可选追问
# =========================
def run_step2_v23():
    if not API_KEY:
        print("❌ 错误：未检测到 API_KEY。")
        return
    if not os.path.exists(LIBRARY_FILE):
        print("❌ 错误：未发现 library.json")
        return

    with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
        library = json.load(f)

    all_pdfs = list(library.keys())
    print("\n📄 论文清单：")
    for i, pdf in enumerate(all_pdfs):
        print(f"  [{i+1}] {pdf}")

    try:
        idx = int(input("\n👇 选择要精读的论文序号: ")) - 1
        target_pdf = all_pdfs[idx]
    except:
        return

    print(f"\n🔍 正在加载并采样：{target_pdf}")
    loader = PyPDFLoader(target_pdf)
    docs = loader.load()

    indices = _pick_indices(docs)
    context_text = _build_context(docs, indices)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
你是一个具备顶级导师素养的“科研硬核审计官”。
你必须解构当前论文的底层物理逻辑，为 GitHub 用户提供一份【完全替代原文阅读】的高浓度研报。

### 🚨 审计准则（内容生产协议）：
1. **全面阅读，极致拆解**：严禁使用一句话带过。必须覆盖：研究冲突、逻辑执行链、核心机制、实验真实性。
2. **拒绝“AI 味”废话**：严禁输出常识性的建议。只输出论文独有的、非显而易见的技术发现。
3. **强制逻辑连贯**：在描述机制时，必须解释“为什么这个设计能解决那个问题”，字数必须丰满。

分页文本如下（只能基于它，不许脑补事实）：
{context_text}

现在返回严格 JSON（value 全中文）：
{{
  "readme_tagline": "一句能击中开发者痛点的技术核心亮点总结",
  "summary": {{
    "problem": "全文深度背景综述：描述研究员在这一技术出现前的痛苦与本质矛盾（不少于 120 字）。",
    "method": "核心逻辑执行链条：第 1 步输入什么？触发了什么变化？第 2 步如何接力？（详述流程，不少于 150 字）。",
    "result": "最终战果与结果意味着什么？（不少于 100 字）",
    "evidence_pages": [1, 2, 3]
  }},
  "table_map": {{
    "pipeline_table": [
      {{
        "stage": "阶段名",
        "input": "输入",
        "core_op": "底层齿轮如何咬合？（详述数据流向的改变，不少于 150 字）",
        "output": "输出",
        "evidence": "[PAGE X] 原句",
        "cost": "隐形代价与牺牲"
      }}
    ],
    "mechanism_table": [
      {{
        "mechanism": "机制名",
        "what_it_does": "原理解析：如何通过算法技巧绕过旧有限制？（不少于 150 字）",
        "why_needed": "解决什么痛点",
        "hidden_cost": "物理上限或失效场景",
        "evidence": "[PAGE X] 原句",
        "how_to_verify": "生活化类比（如：快递分拣、警察断案等）"
      }}
    ],
    "reading_order": ["先看pipeline_table", "再看mechanism_table", "最后进入深度对线"]
  }},
  "audit_cards": [
    {{
      "title": "审计点标题",
      "pain_point": "核心矛盾",
      "evidence": {{ "page": 1, "quote": "原文证据", "interpretation": "通俗解构" }},
      "inference": {{ "assumptions": "推断前提", "impact_range": "影响区间", "quick_test": "1小时验证" }},
      "simple_translation": "范式转移：对比老方法，它在思维上发生了什么跃迁？"
    }}
  ],
  "repro_card": {{
    "known_from_paper": ["原文明确的设置"],
    "mentor_fillins": ["审计官质疑：实验对比是否公平？有没有隐藏的副作用？"],
    "single_gpu_plan": "核心贡献最强的证据是哪张表/哪个指标？",
    "fallback": "边界条件：这个算法在什么极端情况下会彻底失效？"
  }},
  "interactive_battle_guide": {{
    "suggested_queries": [
      {{
        "query": "关于[某逻辑漏洞]的深度质疑提问",
        "reason": "审计动机：为什么这里值得追问",
        "cite_page": 1
      }}
    ]
  }},
  "cited_pages": {list(indices)}
}}
"""

    print("🤖 正在生成“细分精读 + 全要点收敛”JSON ...")
    
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system", 
                "content": "你是一个严谨的论文精读引擎，强制且只输出合法的中文 JSON 对象。注意：如果内容中需要包含双引号，必须使用反斜杠转义（\\\"），绝不能破坏 JSON 结构！"
            },
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"}
    )

    raw = resp.choices[0].message.content
    
    print("\n" + "="*40 + " 大模型原始输出 " + "="*40)
    print(raw)
    print("="*94 + "\n")
    
    data = _parse_json_strict(raw)
    _validate_schema(data)
    data = _parse_json_strict(raw)
    _validate_schema(data)

    followups = _generate_followups_heuristic(data)

    safe_name = re.sub(r'[\\/:*?"<>|]', "_", os.path.basename(target_pdf).replace(".pdf", ""))
    out_name = f"V23_Step2_DeepRead_{safe_name}.md"
    md = _render_md_cn(target_pdf, data, followups)
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n✅ 精读研报已生成：{out_name}")
    print(f"📌 本次采样页码：{[i+1 for i in indices]}\n")

    print("下一步你要做什么？")
    print("  [1] 追问模式A：只基于本论文采样页（深度对线）")
    print("  [2] 追问模式B：接入 Step4 向量库检索追问")
    print("  [3] 退出")
    choice = input("👇 选择：").strip()

    if choice == "1":
        battle_guide = data.get("interactive_battle_guide", {}).get("suggested_queries", [])
        ctx = context_text 
        system_prompt = f"""
你是一位具备顶级学术视野的硬核科研导师。你需要基于给定分页文本回答用户的深度质疑。

回答规则：
1) 结论先行（1-2句直击要害）。
2) 优先寻找原文证据：如果有，必须附上 [PAGE X] 引文（<=60字）进行论证。
3) 【💡关键规则】：如果原文避重就轻或未提供明确答案，你绝不能只说“原文未提供”就结束。你必须：
   - 首先明确声明：“⚠️ 原文未直接提及此原因”。
   - 然后动用你的 AI 领域专业知识储备，补充一段：“🧠 导师推断：[给出你认为最合理的工程/理论解释]”。

分页文本如下：
{ctx}
"""
        print("\n🧩 已进入追问模式 A（只基于本论文采样页，严格证据，不脑补）")
        
        while True:
            print("\n" + "━"*60)
            print("📥 推荐深度对线问题（来自审计建议）：")
            for i, q in enumerate(battle_guide, 1):
                print(f"  [{i}] ⚡ 质疑：{q.get('query', '')}")
                print(f"      └─ 动机: {q.get('reason', '')} (对应 P{q.get('cite_page', '')})")
            
            print("\n  [s4] 🧠 切换到 Mode B (Step4 跨论文检索)")
            print("  [q] 🚪 退出审计系统")
            print("-" * 60)
            
            user_input = input("👉 请输入序号直接对线，或输入你自己的质疑：").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("已退出。")
                break
            if user_input.lower() == 's4':
                _followup_chat_step4()
                break
            
            final_query = user_input
            if user_input.isdigit() and 0 < int(user_input) <= len(battle_guide):
                final_query = battle_guide[int(user_input)-1].get("query", "")
                print(f"\n🙋 你问：{final_query}")
                
            print("🤖 导师正在对线证据库...")
            followup_resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"论文：{target_pdf}\n问题：{final_query}"}
                ],
                temperature=0.1
            )
            print("\n🤖 回答：\n" + followup_resp.choices[0].message.content.strip() + "\n")

    elif choice == "2":
        _followup_chat_step4()
    else:
        print("已退出。")

if __name__ == "__main__":
    run_step2_v23()
