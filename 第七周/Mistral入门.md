# Mistral简介
Mistral有多个模型，其中的**8×7B模型**采用专家（FFN）混合策略对标准的transformer架构进行改造，  在该模型中有8个前向反馈网络被称为“专家experts”。在inference的时候，另一个门控神经网络（router）会首先选择其中两个专家来预测下一个token，然后取其加权平均值来生成下一个token。
这种策略让mistral既有大型模型的性能提升，又有和小型模型相当的推理成本。模型有400多亿个参数，但是在推理时只用到160+亿 。
Mistral另一个实用的特性是**解析json**，当llm集成到更大的软件应用中时，llm能够返回结构化的json格式。
Mistral模型有开源的Mistral 7B和Mistral 8X7B，商用的Mistral small、Mistral medium和Mistral large。
# Overview
模型基于一个由两层组成的transformer块，分别是一个前馈层和一个多头注意力层。为了增加模型容量，复制了N次前馈网络层。使用一个router路由器将每个token映射到顶部的k个前馈层，并忽略其他的层。
![Pasted image 20240807200956](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240807200956.png)
商用的模型中，Mistral large的性能接近于GPT-4，具有多语言处理能力。此外Mistral还有一个embedding模型，用于聚类和分类。
[Le Chat](https://chat.mistral.ai/)：类似于chatgpt，注册即可使用。
本地运行：使用transformers、llama.cpp、ollama等库
使用API：在[La Plateforme](https://console.mistral.ai)中设置API密钥![Pasted image 20240807205354](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240807205354.png)
# Prompting
安装相关库：`pip install mistralai`
## prompt文本
```Python
prompt = """
    You are a bank customer service bot. 
    Your task is to assess customer intent and categorize customer 
    inquiry after <<<>>> into one of the following predefined categories:
    
    card arrival
    change pin
    exchange rate
    country support 
    cancel transfer
    charge dispute
    
    If the text doesn't fit into any of the above categories, 
    classify it as:
    customer service
    
    You will only respond with the predefined category. 
    Do not provide explanations or notes. 
    
    ###
    Here are some examples:
    
    Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card as expected. Could you please provide information about the tracking process for my card, or confirm if there are any indicators to identify if the card has been lost during delivery?
    Category: card arrival
    Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for foreign transactions.
    Category: exchange rate 
    Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and would appreciate any information regarding compatibility and functionality in these regions.
    Category: country support
    Inquiry: Can I get help starting my computer? I am having difficulty starting my computer, and would appreciate your expertise in helping me troubleshoot the issue. 
    Category: customer service
    ###
    
    <<<
    Inquiry: {inquiry}
    >>>
    Category:
"""
```
在这段Prompt文本中，首先使用role-play来为模型提供一个角色-银行客户服务机器人。其次，使用了few-shot learning，可以提高模型的性能。第三，使用哈希或者角括弧等标识符来指定文本不同部分的边界。

该prompt文本的任务是评估客户意图并对客户咨询进行分类，首先给出一些预先定义的种类，并提供了一些例子，通过`format`方法替换`inquiry`来进行查询：
```Python
mistral(
    response.format(
        inquiry="I am inquiring about the availability of your cards in the EU"
    )
)
```
## JSON格式提取信息
```Python
prompt = f"""
Extract information from the following medical notes:
{medical_notes}

Return json format with the following JSON schema: 

{{
        "age": {{
            "type": "integer"
        }},
        "gender": {{
            "type": "string",
            "enum": ["male", "female", "other"]
        }},
        "diagnosis": {{
            "type": "string",
            "enum": ["migraine", "diabetes", "arthritis", "acne"]
        }},
        "weight": {{
            "type": "integer"
        }},
        "smoking": {{
            "type": "string",
            "enum": ["yes", "no"]
        }}
}}
"""
```
在prompt中，给出一个模板，要求模型能够输出json格式文本。
在调用函数时，需要将`is_json`设定为True，以启用JSON模式。
```Python
response = mistral(prompt, is_json=True)
print(response)
```
下面是辅助函数`mistral`的定义：
```Python
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage  
def mistral(user_message,
            model="mistral-small-latest",
            is_json=False):
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    messages = [ChatMessage(role="user", content=user_message)]
    if is_json:
        chat_response = client.chat(
            model=model,
            messages=messages,
            response_format={"type": "json_object"})
    else:
        chat_response = client.chat(
            model=model,
            messages=messages)
    return chat_response.choices[0].message.content
```
首先定义`mistral`的client，使用api key来运行，定义与模型对话的`message`。如果需要返回Json格式，则需要添加`response_format={"type": "json_object"}`。
# Model selecting
Mistral提供五个模型：

| Model          | Endpoint              |
| -------------- | --------------------- |
| Mistral 7B     | open-mistral-7b       |
| Mixtral 8x7B   | open-mixtral-8x7b     |
| Mistral Small  | mistral-small-latest  |
| Mistral Medium | mistral-medium-latest |
| Mistral Large  | mistral-large-latest  |
## Mistral small
mistral small模型适合用于简单任务，快速推理，费用较低，比如用于垃圾邮件分类：
```Python
prompt = """
Classify the following email to determine if it is spam or not.
Only respond with the exact text "Spam" or "Not Spam". 

# Email:
🎉 Urgent! You've Won a $1,000,000 Cash Prize! 
💰 To claim your prize, please click on the link below: 
https://bit.ly/claim-your-prize
"""
mistral(prompt, model="mistral-small-latest")
# 'Spam'
```
## Mistral medium
中型模型适用于中级任务，如语言转换。根据提供的上下文撰写文本（例如，根据购买信息撰写客户服务电子邮件）。
```Python
prompt = """
Compose a welcome email for new customers who have just made 
their first purchase with your product. 
Start by expressing your gratitude for their business, 
and then convey your excitement for having them as a customer. 
Include relevant details about their recent order. 
Sign the email with "The Fun Shop Team".

Order details:
- Customer name: Anna
- Product: hat 
- Estimate date of delivery: Feb. 25, 2024
- Return policy: 30 days
"""
response_medium = mistral(prompt, model="mistral-medium-latest")
```
## Mistral large
大型模型适合需要高级功能、高级推理的复杂任务。
```Python
prompt = """
Calculate the difference in payment dates between the two \
customers whose payment amounts are closest to each other \
in the following dataset. Do not write code.

# dataset: 
'{
  "transaction_id":{"0":"T1001","1":"T1002","2":"T1003","3":"T1004","4":"T1005"},
    "customer_id":{"0":"C001","1":"C002","2":"C003","3":"C002","4":"C001"},
    "payment_amount":{"0":125.5,"1":89.99,"2":120.0,"3":54.3,"4":210.2},
"payment_date":{"0":"2021-10-05","1":"2021-10-06","2":"2021-10-07","3":"2021-10-05","4":"2021-10-08"},
    "payment_status":{"0":"Paid","1":"Unpaid","2":"Paid","3":"Paid","4":"Pending"}
}'
"""
response_small = mistral(prompt, model="mistral-small-latest")
print(response_small)# wrong answer
response_large = mistral(prompt, model="mistral-large-latest")
print(response_large)# right answer
```
小型模型的回答错误，但由于模型的输出是概率性的，所以在多次运行后可能会输出正确结果。
大型模型将问题分为多个步骤，并给出正确的答案。

Mistral large的代码实现也是可行的。

Mistral large能够理解和生成多种语言的文本，可以使用 Mistral 模型进行更多的工作，而不仅仅是将一种语言翻译成另一种语言。
# Function calling
函数调用允许Mistral模型连接外部工具，可以构建特定的用例和实际问题的应用程序。
函数调用步骤：
1. 用户自定义工具tools并使用查询。tool可以是用户自定义的函数，也可以是外部的API。
2. 由模型生成函数参数。模型根据tools和用户查询，可以决定应该使用tools中的哪个函数
3. 用户执行函数，从函数中得到结果。
4. 模型根据现有的资料生成最终答案。
![Pasted image 20240808160115](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240808160115.png)
# RAG from scratch
RAG（Retrieval Augmented Generation）是一个ai框架，结合了大预言模型的信息检索系统的能力，有助于回答问题或生成与外部只是相关的内容。
当用户就内部文件或知识库提出问题时，我们从知识库中检索相关信息。其中，所有的文本前如都存储在一个向量存储区中，这一步骤称为retrieval。在promt中，使用查询和相关信息，模型能够根据相关语境生成输出结果，该步骤成为generation。
## RAG例子
首先使用`BeautifulSoup`来获取文章，然后将文章分块以便更有效识别和检索最相关的信息。
```Python
import requests
from bs4 import BeautifulSoup
import re

response = requests.get(
    "https://www.deeplearning.ai/the-batch/a-roadmap-explores-how-ai-can-detect-and-mitigate-greenhouse-gases/"
)
html_doc = response.text
soup = BeautifulSoup(html_doc, "html.parser")
tag = soup.find("div", re.compile("^prose--styled"))
text = tag.text
print(text)
chunk_size = 512
chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
```
得到文本块后，需要对其进行embedding：
```Python
import os
from mistralai.client import MistralClient
import numpy as np
def get_text_embedding(txt):
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    embeddings_batch_response = client.embeddings(model="mistral-embed", input=txt)
    return embeddings_batch_response.data[0].embedding

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
```
得到嵌入后，通常的做法是将他们存储在适量数据库中，以便高效处理和检索。
```Python
import faiss
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)
```
当用户提出问题时，问题也需要做同样的embedding。
接下来，从矢量数据库中检索文本块：
```Python
D, I = index.search(question_embeddings, k=2)
```
该函数返回的是k个在矢量数据库中与问题最相似的文本块。
将检索到的文本块和问题结合，作为prompt中的上下文信息。
```Python
prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""
response = mistral(prompt)
print(response)
```
## RAG + Function calling
定义一个函数`qa_with_context`
```Python
def qa_with_context(text, question, chunk_size=512):
    # split document into chunks
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    # load into a vector database
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)
    # create embeddings for a question
    question_embeddings = np.array([get_text_embedding(question)])
    # retrieve similar chunks from the vector database
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    # generate response based on the retrieve relevant text chunks

    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    response = mistral(prompt)
    return response
```
将该函数编入到词典中，用于tools构建。
```Python
import functools
names_to_functions = {"qa_with_context": functools.partial(qa_with_context, text=text)}
```
使用一个json模式来概括功能规格，告诉模型tools的功能：
```Python
tools = [
    {
        "type": "function",
        "function": {
            "name": "qa_with_context",
            "description": "Answer user question by retrieving relevant context",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "user question",
                    }
                },
                "required": ["question"],
            },
        },
    },
]
```
将用户问题和tool传入给模型：
```Python
question = """
What are the ways AI can mitigate climate change in transportation?
"""
client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
response = client.chat(
    model="mistral-large-latest",
    messages=[ChatMessage(role="user", content=question)],
    tools=tools,
    tool_choice="any",
)
response
```
其中`tool_choice`有以下选项：
1. `any`：强制模型使用tools中的一种
2. `auto`：让模型自动选择，可以使用tools或者不使用
3. `none`：禁止模型使用tools。
# Chatbot
在本节中将构建一个可视化的对话窗口。
首先引入`panel`，Panel 是一个开源 Python 库，可用于创建仪表盘和应用程序。`pn.extension()`用于加载自定义Javascript和CSS扩展。
```Python
import panel as pn
pn.extension()
```
更新`mistral`函数，新增`user`和`chat_interface`两个参数：
```Python
def run_mistral(contents, user, chat_interface):
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    messages = [ChatMessage(role="user", content=contents)]
    chat_response = client.chat(
        model="mistral-large-latest", 
        messages=messages)
    return chat_response.choices[0].message.content

chat_interface = pn.chat.ChatInterface(
    callback=run_mistral, 
    callback_user="Mistral"
)
chat_interface
```
使用`pn.chat.ChatInterface`定义了一个聊天界面软件，该部件负责处理聊天机器人的所有用户界面和逻辑。
## 上传文件
定义一个文件输入widget以便上传文件，在header中将该部件加入。
```Python
file_input = pn.widgets.FileInput(accept=".txt", value="", height=50)

chat_interface = pn.chat.ChatInterface(
    callback=answer_question,
    callback_user="Mistral",
    header=pn.Row(file_input, "### Upload a text file to chat with it!"),
)
chat_interface.send(
    "Send a message to get a reply from Mistral!", 
    user="System", 
    respond=False
)
chat_interface
```