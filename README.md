# HGCQA: A Benchmark for Complex Question Answering in Chinese History and Geography

**HGCQA (Chinese History and Geography Complex Question Answering)** the first large-scale Chinese CQA benchmark specifically designed for history and geography domains, comprising 10,103 expert annotated question-answer pairs with multi-stage quality validation.
---Research Challenge:
Large Language Models (LLMs) exhibit notable limitations in handling complex question answering (CQA) tasks within the domain of Chinese history and geography, primarily due to knowledge gaps and inadequate reasoning capabilities.

---Key Issues:
Domain-Specific Knowledge Deficits
LLMs lack comprehensive coverage of nuanced historical events, cultural contexts, and geographical specifics unique to China.

---Reasoning Shortcomings
Struggles with multi-hop reasoning (e.g., linking dynastic timelines to regional changes).
Weak performance on comparative (e.g., "Compare the administrative systems of Tang and Ming dynasties") and temporal reasoning tasks.

---Language and Context Barriers
Classical Chinese terms, archaic place names, and culturally embedded metaphors further complicate comprehension.
![NLPCC1](https://github.com/user-attachments/assets/2ebc7de3-5500-471b-89ca-8c0b879db8e3)

---
## 📌 Introduction
Most existing CQA datasets are designed for **open-domain English questions**. In contrast, real-world applications often focus on **vertical domains** in **non-English languages**, especially Chinese.
To address this gap, we introduce **HGCQA**, a benchmark dataset of **10,103 Chinese complex questions**, annotated with:
- Gold-standard answers
- One of 8 complexity types:
  - Bool
  - Comparison
  - Count
  - Difference
  - Intersection
  - Multi-hop
  - Ordinal
  - General

HGCQA is the first benchmark focusing on **Chinese CQA** in history and geography.

---



## Dataset Construction Process
![n30](https://github.com/user-attachments/assets/ffe13017-3c95-4fab-892c-086772bc755a)


## Question Generation Method
![图3-3](https://github.com/user-attachments/assets/fc80f25c-0668-446c-b6b7-219e805520ab)


## 🧠 Dataset Overview
- 🧾 Total questions: `10,103`
- 📚 Source: Chinese Wikipedia (filtered for historical and geographical content)
- 🏷️ Format: JSON
  ```json
  {
    "question": "赵允让的第十三子在位期间的年号是什么？",
    "answer": "治平",
    "type": "Multi-hop"
  },
  {
    "question": "中国西部第四大城市是？",
    "answer": "昆明市",
    "type":"Ordinal"
  },
  {
    "question": "印刷术是否是南宋开始出现的？",
    "answer": "否",
    "type":"Bool"
  },
  {
    "question": "元朝共经历了多少位皇帝？",
    "answer": "11位",
    "type": "Count"
  },
  {
    "question": "两湖平原的西部边界是什么山脉？",
    "answer": "大别山",
    "type":"General"
  },
  {
   "question": "元朝哪位蒙古王公大臣是虔诚的伊斯兰教徒，并且曾经很有机会继承为皇帝？",
   "answer": "安西王阿难答",
   "type":"Intersection"
  },
  {
   "question": "金太宗和金熙宗，谁的在位时间更长?",
   "answer": "金熙宗",
   "type":"Comparison"
  },
  {
   "question": "长江三大矶不包括以下哪个：燕子矶、采石矶、广陵矶？",
    "answer": "广陵矶",
    "type":"Difference"
  }
## Evaluation Metrics and Methods
We adopt exact match (EM) as the
evaluation metric. However, since LLMs often generate answers in descriptive forms while the dataset provides concise answer phrases, direct EM matching becomes challenging. To address this, we employ LLM-based evaluation to better align with model outputs. Based on this, we report both overall and per-type accuracy, providing a comprehensive assessment of LLM performance.
![图1-3](https://github.com/user-attachments/assets/0ffcf7da-da6f-4569-8de1-13a8ad790f88)

  
## 🚀 Baselines and Experimental Results
We evaluate 7 large language models across 3 retrieval methods and 3 reasoning strategies:
Retrieval Methods: BM25, Dense Retrieval (bge-m3), Hybrid Retrieval，They are implemented based on the LangChain framework.
#### General framework with RAG for augmented experiments
![n31](https://github.com/user-attachments/assets/6d783428-466f-46b8-a24e-ad1c194aefce)


## Reasoning Prompts: COT, ThoT, ReAct

## COT Prompt Structure
![图4-4](https://github.com/user-attachments/assets/65da3ec4-b21b-4aac-8e82-32dd7bcbd91f)
ThoT Prompt Structure
![图4-5](https://github.com/user-attachments/assets/197ae2b2-19fe-425b-9f4f-a76594dbdb30)
ReAct Prompt Structure
![图4-6](https://github.com/user-attachments/assets/40c149fd-5b52-4679-8375-c06559480eb1)

LLMs for experiment,The evaluated models are hosted on Baidu Intelligent Cloud Platform and Zhipu AI, with inference performed through their respective APIs.
Llama2-7B
Chatglm2-6B
Llama2-13B
Llama2-70B
Glm3-turbo
ERNIE-3.5
Glm-4


🔍 Best-performing setup:
Model: GLM-4
Retrieval: Hybrid Retrieval
Prompt: ReAct
Best type：Difference 94.60%
Worst type: Ordinal 76.80%
Accuracy: 89.18%

Worst-performing setup 
Model:Llama2-7B
Retrieval:  BM25
Prompt: COT
Best type：General 53.46%
Worst type: Ordinal 32.15%
Accuracy: 36.99%
For full benchmark results, see the paper and results/ directory.
