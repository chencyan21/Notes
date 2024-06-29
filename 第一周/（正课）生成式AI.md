语言模型（ChatGPT）：文字接龙。每次产生答案都是从一个几率分布中取样，答案具有随机性。
每次取样到一个结果后，将结果加入到输入中，再次取样。
考虑过去的对话历史记录：输入不仅有现在的输入，还包括过去的输入
![Pasted image 20240628171417](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240628171417.png)
# ChatGPT流程概述

chatGPT的关键技术：预训练（自监督式学习）
> 一般的监督学习：以中英文翻译为例，给出中英文成对例句，让机器自动寻找函数$f$。缺点在于：给出的资料/数据是有限的。
> chatGPT给出的解决方案：利用网络上的每一段文字来做文字接龙。

chatGPT历史版本：
* GPT-3模型参数大，数据量大，但是只从网络上学习，没有人工介入，使得结果不可控。

![Pasted image 20240628215425](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240628215425.png)
微调finetune：GPT-3通过人工介入的监督式学习，才成为chatGPT。原GPT-3利用网络资料进行学习的过程称为预训练。
自监督式学习：数据集并非人工构造，而是通过某些方法自动生成。
![Pasted image 20240628220925](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240628220925.png)
强化学习：使用PPO算法，人工告诉机器答案好与不好。更适合用于人类不知道答案的好坏的情况。

ChatGPT具体流程：预训练->监督式学习->强化学习
# ChatGPT带来的研究问题
相关研究问题：
1. 如何精确提出需求？
	* 使用特定的prompting。
2. 如何更正错误？修改一个错误后，可能会引发更多的错误。
	* 相关研究[Neural Editing](https://github.com/immortalCO/NeuralEditor)
3. 检测AI生成的相关文字、语音
	* 训练特定模型区分人工和AI生成
4. 是否会泄露相关秘密
	* 相关研究Machine Unlearning，遗忘曾经学习过的东西。

# ChatGPT相关应用-文字冒险
![Pasted image 20240628224412](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240628224412.png)
# （延伸）chatGPT是怎么炼成的
chatGPT的一个类似模型叫做[Instruct GPT](https://arxiv.org/abs/2203.02155)

chatGPT的学习四阶段：
1. 学习文字接龙
2. 人类老师引导文字接龙的方向
3. 模仿人类老师的喜好
4. 用强化学习向模拟老师学习

在阶段一中，GPT可以产生各种答案，在后续的阶段中，需要一步一步引导GPT回答出特定的答案。