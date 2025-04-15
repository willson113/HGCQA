import os
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.llms import BaiduWenxin

# ——— 0. 设置 API Key ———
os.environ["BAIDU_API_KEY"] = "your_apikey"  # 替换为你的文心API key

# ——— 1. 加载问题数据集 HGCQA ———
with open("HGCQA.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ——— 2. 加载本地 bge-m3 模型 ———
embeddings = HuggingFaceEmbeddings(model_name="/data/bge-m3")

# ——— 3. 加载 Chroma 向量库（持久化路径） ———
vectordb = Chroma(
    persist_directory="/data/Chroma_db",  # 向量库目录
    embedding_function=embeddings
)

# ——— 4. 创建 MMR 检索器（提升多样性） ———
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,             # 返回前6个段落
        "lambda_mult": 0.7  # 控制相关性和多样性的权重
    }
)

# ——— 5. 初始化大语言模型（文心一言） ———
llm = BaiduWenxin(model="ernie-bot-turbo", api_key="your_apikey")
tools = "ChromaRetriever"

# ——— 6. 多轮推理提示模板（ReAct 风格） ———
prompt_template = """请你尽量简明扼要地回答以下问题，并根据需要使用以下工具:{tools}

请按照以下格式回答:
问题: 你需要回答的问题
思考: 你应该首先考虑如何处理问题
行动: 你需要采取的行动，应该是[{tools}]
行动输入: 行动所需的输入
观察: 行动的结果
…(这个思考/行动/行动输入/观察过程可以重复零次或多次)

### 特别说明:
1. 如果你在第3轮时仍无法确定答案，请使用反思策略，对已获取的信息进行反思，以检验是否遗漏了任何关键信息或逻辑错误。
2. 如果你在第6轮仍无法确定答案，无论是否完全确定，请给出你推断出的最终答案。你可以明确说明答案的不确定性。
3. 如果你确定可以直接回答问题，请立即给出最终答案，避免不必要的工具调用。
4. 在所有情况下，尽量减少迭代次数，并通过检索补充必要信息。

### 开始!
问题:{imnput}
思考:{fagent_scratchpad}
"""

# ——— 7. 断点续跑机制 ———
output_path = "answer.json"
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    start_idx = len(results)
    print(f"🔁 检测到已有 {start_idx} 条结果，将从第 {start_idx+1} 条继续处理。")
else:
    results = []
    start_idx = 0

# ——— 8. 主循环：语义检索 + 多轮思考 + 实时写入 ———
for idx in range(start_idx, len(data)):
    item = data[idx]
    q = item["question"]
    qtype = item["type"]

    try:
        # 8.1 使用 MMR 检索相关段落（Top 6）
        docs = retriever.get_relevant_documents(q)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 8.2 拼接 Prompt 输入
        prompt = (
            f"段落：\n{context}\n\n" +
            prompt_template.format(
                tools=tools,
                imnput=q,
                fagent_scratchpad=""
            )
        )

        # 8.3 调用大语言模型生成答案
        pred = llm(prompt).strip()

    except Exception as e:
        pred = f"Error: {e}"

    # 8.4 保存结果
    results.append({
        "question": q,
        "answer": pred,
        "type": qtype
    })
    print(f"[{idx+1}/{len(data)}] Q: {q}\nA: {pred}\nType: {qtype}\n")

    # 8.5 实时写入 answer.json（避免崩溃丢失）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ 所有问题处理完毕，答案已写入 answer.json")
