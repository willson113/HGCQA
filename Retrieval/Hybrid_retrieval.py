import os
import json
import signal
import sys
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from zhipuai import ZhipuAI

# 初始化 Zhipu 客户端
client = ZhipuAI(api_key="your_api_key")

# 加载问题数据
with open("HGCQA.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 加载语料
with open("knowledge.txt", "r", encoding="utf-8") as f:
    paras = f.read().split("\n\n")

docs = [Document(page_content=p, metadata={"id": i}) for i, p in enumerate(paras)]

# 初始化检索器
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 6

dense_emb = HuggingFaceEmbeddings(model_name="/data/bge-m3")
dense_vectordb = Chroma.from_documents(docs, dense_emb, persist_directory="./hybrid_chroma")
dense_vectordb.persist()
dense_retriever = dense_vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.7})

# 构造混合检索器
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
)

# 中断保存
output_path = "hybrid_result.json"
results = []
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"已加载已有结果，共 {len(results)} 条")

# 中断信号处理
def save_results():
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已保存结果")

def signal_handler(sig, frame):
    print("  检测到终止信号，保存中...")
    save_results()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 主循环
try:
    for idx in range(len(results), len(data)):
        item = data[idx]
        question = item["question"]
        qtype = item["type"]

        top_docs = hybrid_retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in top_docs])

        first_prompt = (
            f"First Prompt\n"
            f"段落:{context}\n"
            f"问题:{question}\n"
            "逐步引导我理解这个内容，将其分解成易于管理的部分，同时在过程中进行总结和分析。\n"
            "First answer:"
        )

        first_answer = client.chat.completions.create(
            model="glm-3-turbo",
            messages=[
                {"role": "user", "content": first_prompt}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=512,
            stream=False
        ).choices[0].message.content.strip()


        second_prompt = (
            f"Second Prompt\n"
            f"总结和分析:{first_answer}\n"
            f"问题:{question}\n"
            "所以，答案是:"
        )

        final_answer = client.chat.completions.create(
            model="glm-3-turbo",
            messages=[
                {"role": "user", "content": second_prompt}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=256,
            stream=False
        ).choices[0].message.content.strip()


        results.append({
            "question": question,
            "type": qtype,
            "first_prompt_summary": first_answer,
            "final_answer": final_answer
        })

        print(f"\n[{idx+1}/{len(data)}] Q: {question}")
        print(f"First Summary:\n{first_answer}")
        print(f"Answer:\n{final_answer}\n")

        save_results()

except Exception as e:
    print(f"出错：{e}")
    save_results()

finally:
    save_results()
