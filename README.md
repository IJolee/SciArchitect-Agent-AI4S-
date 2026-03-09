SciArchitect-Agent: 双轨制学术文献智能体
1. 项目概述 (Project Overview)

SciArchitect-Agent 是一个基于 LangChain 框架构建的本地化科研文献检索与推理系统，专为处理高密度、逻辑严谨的学术 PDF 论文而设计。旨在解决传统大模型在处理长篇学术文献时易产生的上下文碎片化和推理偏差问题，提升学术文献的阅读和分析效率。

核心架构设计：双引擎架构

检索引擎 (Embedding)：利用 BAAI/bge-m3 模型，实现跨语言文献检索，解决不同语言之间的语义不对称问题。

推理引擎 (LLM)：接入 DeepSeek-V3.2 模型，专注于底层数学推理和机制分析，增强推理的精确性，避免不必要的假设和推理错误。

2. 如何运行 (How to Run)
2.1 系统要求 (System Requirements)

操作系统：Linux / macOS / Windows（我是以windows来做的）

Python 版本：Python 3.7 或更高版本

硬件要求：推荐使用至少 8GB RAM 和 1个 GPU（如 NVIDIA 进行计算加速）来提高处理速度和性能。

2.2 环境配置 (Environment Setup)

克隆项目仓库：

进入你希望存放该项目的文件夹，在命令行中执行以下命令来克隆仓库：

git clone https://github.com/your-username/SciArchitect-Agent.git
cd SciArchitect-Agent

安装 Python 依赖：

确保你已经安装了 Python 3.7+，并且安装了 pip 包管理工具。如果没有安装，可以参考 Python 官方网站
 进行安装。

接下来，你需要安装项目所需的所有依赖包。在项目根目录下执行以下命令：

pip install -r requirements.txt

requirements.txt 文件中列出了所有必要的 Python 库，确保在安装时下载并安装这些库。

2.3 环境变量配置 (Environment Variables Setup)

为了让系统顺利运行，你需要提供 API 密钥。在项目根目录下创建一个 .env 文件，并填入你的 API 密钥：

在项目根目录创建 .env 文件（如果没有的话），并填入：

SILICONFLOW_API_KEY=sk-你的API密钥

请确保 API 密钥是有效的。如果你没有密钥，可能需要注册并获取一个。

这个 API 密钥将用于访问 硅基流动 (SiliconFlow) API。

确保 .env 文件被列在 .gitignore 文件中，以免将敏感信息上传到 GitHub。

2.4 依赖包安装 (Install Dependencies)

项目依赖了多个 Python 库，请确保所有依赖都已安装。你可以通过以下命令来安装它们：

pip install langchain langchain-chroma langchain-openai pypdf python-dotenv

这些依赖包括：

LangChain：用于实现文献检索和推理。

Chroma：用于存储和检索文献向量数据。

OpenAI：用于调用 OpenAI 的 GPT 模型。

PyPDF：用于从 PDF 中提取文本。

python-dotenv：用于加载 .env 文件中的环境变量。

2.5 文献准备 (Preparing the Papers)

将你需要分析的 PDF 文献 文件放置在项目根目录下。这些文献将被自动扫描并用于后续的处理和分析。

2.6 运行流程 (Execution Pipeline)

系统分为多个阶段，每个阶段负责不同的任务。请按照以下顺序执行每个阶段：

Step 1: 图书管理员 (Librarian) - 扫描与清洗
python step1_librarian.py

功能：扫描项目根目录下的 PDF 文件，提取文本内容并生成初步的元数据。

输出：生成 library.json 文件，其中包含每篇论文的元数据。

运行效果截图：


Step 2: 学术教授 (Professor) - 结构化元数据提取
python step2_professor.py

功能：对 Step 1 中提取的无序文本进行学术分类，并生成结构化的元数据文件。

输出：更新 library.json，为文献添加更多详细的分类信息，如领域、任务等。

运行效果截图：


Step 3: 知识库构建 (Update Chat DB) - 向量化与持久化
python step3_update_chat_db.py

功能：对文献内容进行向量化处理，并将处理后的文献数据保存到本地数据库（Chroma）。

输出：将处理后的向量数据存储在 brain_db 目录中。

注意：此步骤可能耗时较长，特别是在处理大规模文献数据时。

运行效果截图：


Step 4: 对话终端 (Chat Assistant) - 深度学术推演
python step4_chat_assistant.py

功能：启动命令行交互界面，接收用户的自然语言问题，并基于文献内容提供精准的学术推理回答。

输出：根据用户输入的学术问题，系统将从本地数据库中检索相关文献并生成答复。

运行效果截图：


Step 5: 研报生成 (Review Writer) - 自动生成文献综述报告 (可选)
python step5_review_writer.py

功能：根据 Step 4 中的问答结果，生成全局主题维度的文献综述报告。

输出：生成一份包含关键论点、实验对比数据与结论的 Markdown 格式报告。

运行效果截图：


3. 核心模块解析 (Step 1 - Step 5)
Step 1: 图书管理员 (Librarian) - 扫描与清洗

功能特点：负责物理文件的 I/O 操作。自动遍历当前目录下的所有 PDF 文件，提取基础文本内容。

Step 2: 学术教授 (Professor) - 结构化打标

功能特点：对 Step 1 提取的无序文本进行高维度的学术分类与特征提取，并生成轻量级的 library.json 元数据配置文件。

Step 3: 知识库构建 (Update Chat DB) - 向量化与持久化

功能特点：系统的“视神经”构建模块。将文档进行多级切片，调用 BGE-M3 模型生成高维向量，并存储至本地 Chroma 数据库 (brain_db)。

Step 4: 对话终端 (Chat Assistant) - 深度学术推演

功能特点：系统的“大脑”。基于终端的命令行交互界面，接收用户的自然语言质询。

Step 5: 研报生成 (Review Writer) - 自动化总结

功能特点：在 Step 4 单轮问答的基础上，进一步支持全局主题维度的文献综述撰写。

4. 注意事项与安全规范 (Precautions)

API 密钥管理：请务必使用 .env 文件管理 API 密钥，避免将密钥硬编码到代码中。

仓库数据隔离：生成的 brain_db 和 docstore_data 会占用大量磁盘空间，且包含敏感文献数据，请确保 .gitignore 忽略这些文件。

网络与模型选用：本系统依赖于特定的模型（BAAI/bge-m3），确保使用正确的模型进行文献分析。

5. 项目可开发性与未来演进 (Future Extensibility)

引入 GraphRAG (知识图谱检索)

跨模态文档解析 (Vision-Language Parsing)

多代理协作 (Multi-Agent System)

工程化前端部署 (Web GUI)

6. 项目许可证 (License)

本项目采用 MIT License，你可以自由使用、修改和分发代码，但请保留原始作者的版权声明和许可证。

结语

SciArchitect-Agent 旨在为学术研究人员提供一个智能、高效的文献分析工具。通过本项目，你可以轻松地从海量文献中提取关键信息，并获得深入的学术分析，助力科研创新。