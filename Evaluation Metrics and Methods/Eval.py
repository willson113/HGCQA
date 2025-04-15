import json
import signal
import sys
from zhipuai import ZhipuAI

# 初始化客户端
client = ZhipuAI(api_key="a91e7a400a15d12690b67227bf423934.y0jjHGrcHI6sCQ6i")

# 读取标准答案和模型答案文件
with open('all.json', 'r', encoding='utf-8') as f:
    standard_data = json.load(f)

with open('gml-3-COT.json', 'r', encoding='utf-8') as f:
    model_data = json.load(f)

# 确保标准答案和模型答案数量一致
assert len(standard_data) == len(model_data), "标准答案和模型答案数量不一致"

results = []


def save_results():
    """将结果保存到 json 文件"""
    with open('ACC-glm3-cot.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def signal_handler(sig, frame):
    """处理终止信号"""
    print("检测到终止信号，保存结果中...")
    save_results()
    sys.exit(0)


# 绑定信号处理函数
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    # 遍历数据集并调用API进行评估
    for std_item, mdl_item in zip(standard_data, model_data):
        question = std_item['question']
        standard_answer = std_item['answer']
        model_answer = mdl_item['answer']

        # 构建评估提示
        prompt = (
            f"问题：{question}\n"
            f"标准答案：{standard_answer}\n"
            f"模型答案：{model_answer}\n"
            "根据所给问题，判断上述模型答案是否与标准答案一致（结合问题，考察模型答案表达的意思是否与标准答案相同）？注意仔细比对标准答案和模型答案，只回答“是”或“否”。"
        )

        try:
            # 调用 API
            response = client.chat.completions.create(
                model="glm-3-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个答案评估专家，你的任务是根据给定的问题、标准答案和模型答案，判断模型答案是否正确，只能回答是或否，不要其他的多余输出。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                top_p=0.7,
                temperature=0.01,
                max_tokens=10,
                stream=False  # 设置为 False 以获取完整的响应
            )

            # 提取回答
            result = response.choices[0].message.content.strip()

            # 记录结果
            results.append({
                "question": question,
                "standard_answer": standard_answer,
                "model_answer": model_answer,
                "result": result
            })

            # 打印每个调用的结果
            print(f"Question: {question}")
            print(f"Standard Answer: {standard_answer}")
            print(f"Model Answer: {model_answer}")
            print(f"Result: {result}")

        except Exception as e:
            # 记录错误信息并跳过该问题
            print(f"处理问题时出错: {e}")
            results.append({
                "question": question,
                "standard_answer": standard_answer,
                "model_answer": model_answer,
                "result": "Error occurred",
                "error": str(e)
            })

        # 每次处理完一个问题后立即保存结果
        save_results()

except Exception as e:
    print(f"程序出现错误: {e}")
    save_results()  # 在异常情况下也保存结果
finally:
    save_results()  # 无论是否出现异常都保存结果
