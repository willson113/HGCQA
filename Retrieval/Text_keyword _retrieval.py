import os
import json
from whoosh import index, fields, qparser
from whoosh.analysis import StemmingAnalyzer
from langchain.llms import BaiduWenxin

#环境变量
os.environ["BAIDU_API_KEY"] = "your_apikey"

#加载 HGCQA.json 
with open("HGCQA.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 加载段落语料
def load_paragraphs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split("\n\n")

paras = load_paragraphs("/data/wikidata/knowledge.txt")

# 构建或加载 Whoosh 索引
INDEX_DIR = "bm25_index"
schema = fields.Schema(
    id=fields.ID(stored=True, unique=True),
    content=fields.TEXT(stored=True, analyzer=StemmingAnalyzer())
)

if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)
    ix = index.create_in(INDEX_DIR, schema)
    writer = ix.writer()
    for i, para in enumerate(paras):
        writer.add_document(id=str(i), content=para)
    writer.commit()
    print("已创建并写入 Whoosh 索引")
else:
    ix = index.open_dir(INDEX_DIR)
    print("已加载已有 Whoosh 索引")

#构建 LLM & Prompt 模板
llm = BaiduWenxin(model="ernie-bot-turbo", api_key="your_apikey")
prompt_template = (
    "段落：\n{paragraphs}\n\n"
    "根据以上的背景知识，让我们一步一步来思考，回答下面的问题：\n"
    "问题：{question}\n"
    "所以，答案是："
)

#  断点续跑 & 初始化 results 
output_path = "answer.json"
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    start_idx = len(results)
    print(f"检测到已有 {start_idx} 条结果，将从第 {start_idx+1} 条继续处理。")
else:
    results = []
    start_idx = 0

# 主循环：检索 + 问答 + 实时保存 
qp = qparser.QueryParser("content", schema=ix.schema)
for idx in range(start_idx, len(data)):
    item = data[idx]
    q = item["question"]
    qtype = item["type"]

    #  BM25 检索 top 6 段落
    query = qp.parse(q)
    with ix.searcher() as searcher:
        hits = searcher.search(query, limit=6)
        retrieved = [hit["content"] for hit in hits]

    paragraphs_str = "\n\n".join(retrieved)

    #调用 LLM
    prompt = prompt_template.format(paragraphs=paragraphs_str, question=q)
    try:
        pred = llm(prompt).strip()
    except Exception as e:
        pred = f"Error: {e}"

    # 保存结果到列表
    results.append({
        "question": q,
        "answer": pred,
        "type": qtype
    })

    print(f"[{idx+1}/{len(data)}] Q: {q}\n    A: {pred}\n    Type: {qtype}\n")

    #  实时写入 answer.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("全部处理完毕，结果已写入 answer.json")
