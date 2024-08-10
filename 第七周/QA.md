常见的问答任务模型：
1. 抽取式extractive：从给定的上下文中提取答案
2. 抽象式abstractive：利用正确回答问题的上下文生成答案。
# 加载数据集
此次任务中使用的数据集市SQuAD数据集的较小子集。
使用`train_test_spilt`方法将数据集拆分为train和test：
```Python
from datasets import load_dataset
squad = load_dataset("squad", split="train[:5000]")
squad = squad.train_test_split(test_size=0.2)
```
squad数据集字段如下：
```
squad:
- train:
	- answer: dict 答案token和答案文本的起始位置
		- answer_start: list[int]
		- text: list[str]
	- context:str 模型需要从中提取答案的背景信息
	- id: str
	- question: str 模型应该回答的问题
	- title: str
- test
```
# Preprocess
同文本分类，加载DistilBERT分词器处理问题和context：
```Python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
...
tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```
数据集中的某些样本具有很长的context，context的长度超过了模型的最大输入长度，需要使用`truncation="only_second"`来截断context。
> `truncation="only_second"`：
> `only_second`是指：当输入包含两个序列时（问答任务中通常是一个问题+一个context），**仅截断第二个序列。**（也就是context部分）

设置`return_offset_mapping=True`来将答案的开始和结束位置映射到原始的context上。
有了映射后，可以寻找到答案的开始和结束标记。
```Python
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
	...
```
# Train
与文本分类相似，加载模型，定义超参数，将训练相关的东西一起传给trainer，然后微调
```Python
training_args = TrainingArguments(...)
trainer = Trainer(...)
trainer.train()
```
# Inference
同文本分类，最简单的方式便是使用`pipeline`：
```Python
from transformers import pipeline
question_answerer = pipeline("question-answering", model="./my_awesome_qa_model/checkpoint-750")
question_answerer(question=question, context=context)
```
如果使用传统方式：
```Python
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("./my_awesome_qa_model/checkpoint-750")
inputs = tokenizer(question, context, return_tensors="pt")
model = AutoModelForQuestionAnswering.from_pretrained("./my_awesome_qa_model/checkpoint-750")
with torch.no_grad():
    outputs = model(**inputs)
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)
```
二者的输出结果都是一样的：
![Pasted image 20240810130119](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810130119.png)
![Pasted image 20240810130128](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810130128.png)