# 对大语言模型的期待
对大语言模型有两种不同的期待：
1. 成为专才，解决某一个特定任务；
2. 成为通才，无论什么任务都能处理。

**Prompt**：用户操控大语言模型执行不同任务的指令
两种方式的好处：
1. 专才：在单一任务上一般比通才的效果更好；
2. 只要重新涉及Prompt就可以快速开发新功能，不用编写代码。

# 方式一：成为专才
## Finetuning
将预训练模型中的参数当作初始参数，使用梯度下降来更新参数，使得新模型能快速达到理想效果。![Pasted image 20240708202448](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240708202448.png)
## Adapter
在新的语言模型中新建一层layer作为adapter，更新参数时，预训练语言模型中的参数保持不变，只更新adapter中的参数。![Pasted image 20240708203047](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240708203047.png)
# 方法二：成为通才
GPT没有像Bert一样进行微调参数的可能原因：
1. 一开始OpenAI有较高期待，让GPT系列比专才更强
2. Bert将微调相关的路线覆盖完全了，只好另辟蹊径。
## 上下文学习In-context Learning
![Pasted image 20240709144359](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709144359.png)告诉模型当前任务是做情感分析，将需要分析的句子以及一些例子一起提供给模型，希望模型能够通过例子学习到这种范式，进而输出正确的结果。
但是论文[Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)提出了相关问题：语言模型能否从这些例子中学习？
**反例一**：故意给出错误标注，但是从结果来看并没有受到太大影响
![Pasted image 20240709152848](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709152848.png)![Pasted image 20240709152858](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709152858.png)**反例二**：给一些无关的输入，结果也同反例一类似。
![Pasted image 20240709152951](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709152951.png)![Pasted image 20240709152956](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709152956.png)
**结论**：
> In-context Learning这些例子并不是让语言模型做学习，而是让模型知道当前该执行什么任务。模型原本就有执行这些任务的能力，所需要的只是唤醒模型的相关能力。

![Pasted image 20240709162148](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709162148.png)图中，机器在模型大到一定程度时，才会从范例中学习，而且越是大型的模型，在给定错误的范例时，受到的影响越大。
## 指令微调Instruction-tuning
![Pasted image 20240709171533](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709171533.png)训练时给机器xx的指令，让模型执行xx的任务，测试时便可以给模型yy的指令，让模型完成yy任务。
具体步骤：先收集大量的自然语言处理任务![Pasted image 20240709171810](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709171810.png)然后将这些任务改写成指令传递给机器让模型学习![Pasted image 20240709171937](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709171937.png)
## CoT思维链
给机器范例的同时，给机器推论过程和答案。
### Least-to-most prompting
![Pasted image 20240709184204](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709184204.png)先将问题传给语言模型拆解解题步骤，然后分步传给语言模型进行解答。
## 用机器来寻找Prompt
### 使用Soft prompt
![Pasted image 20240709190000](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709190000.png)
给机器的指令是一堆可训练的连续向量参数。可以将soft prompt看作是adapter中的特例。
### 使用增强学习
![Pasted image 20240709190206](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709190206.png)
### 使用LM来寻找prompt
![Pasted image 20240709190243](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709190243.png)给模型范例，让模型自己给出相应的prompt。（cot中的相应触发语句）
![Pasted image 20240709190433](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240709190433.png)


