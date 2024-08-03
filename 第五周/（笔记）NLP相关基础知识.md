生成词向量的方法，主要有两类：矩阵奇异值分解方法（SVD）和基于迭代的方法（Word2vec）。Word2Vec有两种主要模型架构：CBOW和Skip-gram。
# one-hot缺点
one-hot不能准确表达不同词之间的相似度。
对于任意向量$x,y$，他们的余弦相似度为：$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1]$$
由于任意两个不同词的one-hot之间的余弦相似度都为0，所以one-hot不能编码词之间的相似性。
# 分布式语义
一个词的含义将由出现在它附近的词来提供。当一个单词`w`出现在一个文本中时，其上下文/context是一组单词出现在附近（固定大小的窗口内）。
词向量（word vector/word embedding/word representation）：我们将为每个单词构建一个密集向量，使其与出现在类似语境中的单词向量相似。词向量属于分布式表示，而非局部表示。
# Word2Vec
Word2Vec是一个学习词向量的框架。
idea:
* 有大量的文本（"正文"）Corpus
* 固定Vocab中的每个词都由一个向量表示
* 查看文本中每个位置t的中心词c和上下文（"外部"）词o
* 利用c和o的词向量的相似性计算出给定c的o的概率（反之亦然）
* 不断调整词向量，使这一概率最大化

对于每个位置t=1，...，T，给定中心词$w_j$，在固定大小$m$的窗口内预测上下文词：$$\textrm{Likelihood}=L(\theta)=\prod_{t=1}^T\prod_{-m\leq j \leq m}P(w_{t+j}|w_t;\theta)$$ $\theta$表示需要优化的所有变量。
目标函数loss function是（平均）负对数似然：$$J(\theta)=-\frac{1}{T}=-\frac{1}{T}\sum_{t=1}^{T}\sum_{-m\leq j \leq m}log P(w_{t+j}|w_t,\theta)$$
word2vec工具包含两个模型，即跳元模型（skip-gram）和连续词袋（CBOW）。
# Skip-gram
在给定中心词的情况下生成周围上下文词。
在skip-gram中，每个词都有两个d维向量表示，用于计算条件概率。对于词典中索引为i的任何词，分别用$\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示其用作**中心词**和**上下文词**时的两个向量。

给定中心词$w_c$（词典中的索引$c$），生成任何上下文$w_o$的条件概率可以通过对向量点积的softmax操作来建模：$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}$$其中词表索引集$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$。给定长度为T的文本序列，其中时间步t处的词表示为$w^{(t)}$。
<font color='red'>此处计算时包含了整个词表大小一项多的项的求和，计算的成本是巨大的，为降低计算复杂度，需使用近似训练方法。</font>
假设上下文词是在给定任何中心词的情况下独立生成的。对于上下文窗口m，skip-gram模型的似然函数是在给定任何中心词的情况下生成所有上下文词的概率：$$\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)})$$
训练时最小化以下损失函数：$$- \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)})$$
其对数条件概率为：
$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right)$$通过微分计算：$$\begin{split}\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j\end{aligned}\end{split}$$
# CBOW
CBOW连续词袋模型假设中心词是基于其在文本序列中的周围上下文词生成的。由于CBOW存在多个上下文词，因此在计算条件概率时对这些上下文词向量进行平均。
对于词典中索引i的任意词，分别用$\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示用作*上下文词和中心词*（**与Skip-gram模型相反**）。给定上下文词$w_{o_1}, \ldots, w_{o_{2m}}$（在词表中索引是$o_1, \ldots, o_{2m}$）生成任意中心词$w_c$的条件概率为：$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}$$
令$\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$和$\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$，式子简化为$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}$$
连续词袋模型的最大似然估计等价于最小化以下损失函数：$-\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)})$即$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right)$$
通过微分可得$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right)$$
