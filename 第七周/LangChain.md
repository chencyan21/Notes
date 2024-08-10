# Introduction
LangChain是一个开源的用于构建LLM应用程序的框架。有两种不同的软件包，分别是Python软件包和JavaScript软件包。专注于组合模块化。
后续探讨方向：
1. 模型
2. prompts
3. 索引indexes
4. chains，端到端使用案例
5. agents，将模型作为推理引擎。
# Models, Prompts and Parsers
模型指的是支撑语言分析程序的语言模型，Prompt是指创建输入的样式，以便将其传递给模型，解析器包含将这些模型的输出结果并将其解析为更有条理的格式。
辅助函数`get_completion`用于与模型对话：
```Python
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]
```
输出解析：
在提示模板中要求输出格式为JSON
```Python
customer_review = """"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)
messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print(response.content)
```
### 使用LangChain API
模型构建：
```Python
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(temperature=0.0, model=llm_model)
chat
```
Prompts构建：
首先构建template：（注意template中没有"f"字符串）
```Python
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template_string)
```
使用`prompt_template.messages[0].prompt.input_variables`可以检测到template中的变量
```Python
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
```
模型对话：
```Python
# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)
print(customer_response.content)
```
 输出解析：
从LangChain中导入ResponseSchema和StructuredOutputParser，通过指定返回的JSON格式来要求LangChain
```Python
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
...
response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
```
`format_instructions`对LLM有一套精确的指令，可以让LLM生成指定的内容格式以便解析器可以解析。
在与模型对话后，将得到的对话使用`output_parser`解析出来。
```Python
output_dict = output_parser.parse(response.content)
```
# Memory
当与模型对话时，正常情况下它们无法记住之前的对话，因此本节探讨如何记住前面的对话内容并能将其输入到语言模型中。
```Python
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```
使用`memory`存储记忆，创建一个`ConversationChain`对话链，通过predict函数与模型对话：
```Python
conversation.predict(input="Hi, my name is Andrew")
```
这种存储记忆的方式是将所有的记忆都存储在`memory`中。
接下来是另一种方式，`ConversationBufferMemory`保留窗口记忆，也就是仅保留最后若干轮对话消息。
```Python
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.load_memory_variables({})
# {'history': 'Human: Not much, just hanging\nAI: Cool'}
```
参数`k=1`表示只记住最后一轮对话。该方法可以防止memory存储量随着对话的进行而无线增长。
使用`ConversationTokenBufferMemory`将限制保存在记忆存储的tokens数量，
```Python
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
memory.load_memory_variables({})
# {'history': 'AI: Amazing!\nHuman: Backpropagation is what?\nAI: Beautiful!\nHuman: Chatbots are what?\nAI: Charming!'}
```
最后一种是`ConversationSummaryBufferMemory`方式，让LLM为所有的历史消息生成摘要，在记忆中存储历史消息的摘要。
```Python
# create a long string
schedule = "..."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
```
# Chains
Chain将LLM和Prompt结合在一起，该模块还可以进行组合，对文本或者其他数据按顺序进行操作。
```Python
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)
```
通过`run`方法来格式化prompt，并将prompt传给LLM。
## Sequential Chains
Sequential Chains是另一种类型的链。这个想法是将多个链组合在一起，其中一个链的输出是下一个链的输入。
有两种类型的顺序链:
1. `SimpleSequentialChain`:单一的输入/输出
2. `SequentialChain`:多个输入/输出
```Python
llm = ChatOpenAI(temperature=0.9, model=llm_model)
# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)
# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )
```
当存在多个输入/输出时，也可以使用Sequential Chains，要求每个chain的输出`output_key`必须对应到后续使用的变量
```Python

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )

# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)
```
![Pasted image 20240809201647](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240809201647.png)
![Pasted image 20240809201709](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240809201709.png)
## Router Chain
一种更复杂的操作是：根据输入的内容route到某条chain来处理，如果有多条subchains，每条subchain专门负责处理某种特定类型的输入，这种情况下可以使用router chain。
![Pasted image 20240809201759](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240809201759.png)
```Python
physics_template = """...{input}"""
math_template = """...{input}"""
history_template = """...{input}"""
computerscience_template = """ ...{input}"""
prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

```
在router中，可以给每个模板起名字，给出描述，这些信息将传递给router chain，router chain将根据这些信息来决定什么时候该使用哪条子链。除了设计好的subchain外，还需要一个默认chain，默认chain用于router找不到合适的subchain时，用来备用的一条chain。
```Python
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
```
构建router chain：
```Python
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
```
# Question and Answer
文档问答是一种常见的用LLM构建的复杂应用程序。
首先初始化一个加载器`CSVLoader`，并为其指定文件路径。
```Python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
```
调用文档加载器，传入向量存储：
```Python
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
```
接着定义一个query，用`index.query`生成一个响应，并传入`query`。将得到的结果用`markdown`展示便得到结果：
![Pasted image 20240810151015](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810151015.png)

Embeddings：可以利用embedding从文档中找出与问题相似的文本片段，一起传递给语言模型来帮助回答问题。
VectorDatabase：向量数据库是一种存储方法，可以存储在embedding中创建的vector。往VectorDatabase中新建数据的方式：将文档拆分成块/chunks，每块生成embedding，然后把embedding和原始块一起存储。

在开始得到的loader后，使用`OpenAIEmbeddings`来嵌入：
```Python
from langchain.embeddings import OpenAIEmbeddings
docs = loader.load()
embeddings = OpenAIEmbeddings()
```
通过在向量存储器上调用`from_documents`方法来将所有的文本片段embedding：
```Python
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)
```
## Query查询
首先从向量存储器创建一个`retriever`，该api定义了一个接受查询内容并返回相似文档的方法。使用LangChain封装：
```Python
retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
response = qa_stuff.run(query)
```
# Evaluation
本节探讨如何评估基于LLM的应用程序。
首先定义之前的内容：
```Python
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
```
导入`QAGenerateChain`，它可以读取文档并从每个文档中生成一组问题和答案，因为需要借助语言模型，所以需要通过传入`ChatOpenAI`来创建。
```Python
from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
```
使用`langchain.debug = True`可以输出更多细节，但是需要人工检索错误或细节。

使用语言模型来判断：
导入`QAEvalChain`，使用语言模型来帮助评估：
```Python
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```
# Agents
agents通常指的是独立的语言模型实例或子程序，它们能够在给定任务中模拟人类对话、完成特定任务，或者在某个环境中自主运作。这些agents可以根据输入生成自然语言响应，并且在一些高级实现中，它们可以执行更加复杂的任务，比如搜索信息、调取数据库、执行代码等。
加载相关工具：
```Python
tools = load_tools(["llm-math","wikipedia"], llm=llm)
```
1. `llm-math`：结合语言模型和计算器来解决数学问题
2. `wikipedia`：一个连接到维基百科api的程序，允许用查询维基百科上的内容并返回搜索结果。
初始化代理：
```Python
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
```
调用agent对话：
```Python
agent("What is the 25% of 300?")
```
## Python agent
创建一个python代理：
```Python
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
```
`PythonREPLTool`是与代码相关的工具。
