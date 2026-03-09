import os
import glob
import json
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader

# === 配置 ===
API_KEY = "sk-hvwxamrlaazoupkruqugfuynjieqgqploofhhliplhyeppik"
BASE_URL = "https://api.siliconflow.cn/v1"
LIBRARY_FILE = "library.json"
# ===========

def scan_and_categorize():
    print("=== 📚 启动 AI 图书管理员 ===")
    
    # 读取旧目录
    library = {}
    if os.path.exists(LIBRARY_FILE):
        with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
            library = json.load(f)

    # 扫描新文件
    pdf_files = glob.glob("*.pdf")
    new_files = [f for f in pdf_files if f not in library]
    
    if not new_files:
        print("🎉 所有论文都已归档！")
        return

    print(f"🔍 发现 {len(new_files)} 篇新论文，正在分类...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for i, pdf in enumerate(new_files):
        print(f"   [{i+1}/{len(new_files)}] 分析: {pdf} ...")
        try:
            # 只读第1页
            loader = PyPDFLoader(pdf)
            first_page = loader.load()[0].page_content[:1500]
            
            prompt = f"""
            阅读这篇论文的首页：
            {first_page}
            
            请提取关键元数据，并以 **纯 JSON 格式** 输出（不要 Markdown）：
            {{
                "title_cn": "中文标题",
                "category": "领域 (CV/NLP/RL...)",
                "task": "具体任务 (如: 目标检测, 文本摘要)",
                "method": "核心方法名 (如: Transformer, Diffusion)",
                "dataset": "使用了哪些数据集 (如: COCO, ImageNet)",
                "metrics": "主要评价指标 (如: mAP, BLEU)",
                "status": "初判价值 (High/Medium/Low)"
            }}
            """
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            library[pdf] = data
            print(f"      🏷️ [{data['category']}] {data['title_cn']}")

        except Exception as e:
            print(f"      ❌ 失败: {e}")

    # 保存
    with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 归档完成！目录已存入 {LIBRARY_FILE}")

if __name__ == "__main__":
    scan_and_categorize()