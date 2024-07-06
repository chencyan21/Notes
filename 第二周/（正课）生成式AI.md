# 机器学习基本概念
机器学习$\approx$机器自动寻找一个函数

根据函数的输出分为以下两类：
1. 回归：函数的输出是一个数值
2. 分类：函数的输出是一个类别
	* ChatGPT将生成式学习拆分成多个分类问题

找出函数的三个步骤：
1. 设定范围
	* 找出候选函数的集合-Model。深度学习中类神经网络的结构（例如CNN、RNN、Transformer等）指的就是不同的候选函数集合。
2. 设定标准
	* 设定评价函数好坏的标准-Loss。半监督学习中，两个没有标注的数据如果通过某种方法可以判定为相似后，那么它们的函数输出结果也应该是相同的，可以将二者的函数结果差值作为loss计算。即$$L(f_1)=输出距离正确答案+长得像的宝可梦差距$$
3. 达成目标
	* 找出最好的函数-Optimization。数学式为$$f^*=arg\mathop{\min_{f\in\mathcal{H}}} L(f)$$相关方法：梯度下降、反向传播
![Pasted image 20240702112510](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240702112510.png)
训练数据集数据越少，越容易发生过拟合。
![Pasted image 20240702164316](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240702164316.png)
残差连接和批量归一化等方法可以减少过拟合现象的发生。

# 生成式学习策略
生成式学习有两种策略：各个击破和一次到位。

生成式学习可以生成有结构的复杂物件：
* 语句：由token构成。中文的token就是字，英文的token是词组。
* 图片：由像素组成。
* 语音：16k取样频率，每秒有16000个取样点
> 生成影片：[Imagen Video](https://arxiv.org/abs/2210.02303)
> 生成语音：[InstructTTS](https://arxiv.org/abs/2301.13662)（不同的语气风格）
> 生成声音：[Text-to-audio](https://arxiv.org/abs/2301.12503)（不限于人的声音），使用chatGPT描述具体声音，然后将描述文本投入模型中生成

策略一-各个击破：Autoregressive (AR) model
策略二-一次到位：Non-autoregressive (NAR) model
![Pasted image 20240703163739](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240703163739.png)
各个击破适用于文本生成等，质量更高，而一次到位适合用于影像生成，速度更快，因为影像中每一帧就有一个图片，涉及到多个像素点，使用各个击破会耗费大量时间。
## 策略相互结合
将两种策略互相结合，以语音合成为例：首先使用各个击破来合成中间数据，然后使用一次到位合成最终结果
![Pasted image 20240703165557](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240703165557.png)
# AI工具
NewBing与ChatGPT不同，NewBing会在网络上搜寻资料，是否进行搜寻是由机器决定。
WebGPT：使用搜索引擎的GPT
ToolFormer：使用多种不同的工具，例如搜索、计算器、翻译器
# Brief Introduction of Deep Learning
深度学习大致历史：
1958：感知机（线性模型）
1959：感知机存在缺陷
1980s：多层感知机（与DNN并没有太大区别）
1986：反向传播
1989：1个隐藏层便已经足够
2006：RBM（受限玻尔兹曼机）
## 网络结构
深度学习中的deep=many hidden layers（三层以上）
AlexNet（2012）-VGG（2014）-GoogleNet（2014）-ResNet（2015）

![Pasted image 20240705155803](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705155803.png)
$Input\times Weight+Bias$然后通过激活函数$\sigma$得到的输出output便是下一层的input。

在最终output层之前的部分，可以看作是一个特征提取器（feature extractor），提取好的特征，可以通过一层多分类器便可以进行最终的分类。这个多分类器/输出层一般是softmax
![Pasted image 20240705163234](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705163234.png)
问题：
1. 中间的隐藏层要有多少层？每层的神经元个数为多少个？（网络如何设计）
	难以决定，一般是直觉+尝试
2. 网络结构能否被自动决定？
	Evolutionary Artificial Neural Networks
3. 能否自己设计网络？
	可以，例如CNN
## Loss和梯度下降
**LOSS可以定义网络的好坏**：
![Pasted image 20240705212038](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705212038.png)
将每个数据的loss累计后便是总loss，需要寻求方法（梯度下降）来最小化loss。![Pasted image 20240705212123](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705212123.png)
首先随机生成一组参数，然后使用梯度下降方法更新参数，直到loss达到限定值。![Pasted image 20240705212454](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705212454.png)
## 反向传播
反向传播：一种更有效率的方式计算网络中微分。![Pasted image 20240705212841](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705212841.png)
## 网络是否更深更好？
![Pasted image 20240705213005](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705213005.png)普遍性原理：任何连续的函数$f$都能被只有一个隐藏层的网络实现（隐藏单元的数量够多）

# Gradient Descent
优化问题：$$\theta^*=\arg \mathop{\min_{\theta}}L(\theta)$$假设$\theta$有两个参数$\theta_1,\theta_2$，随机选取一个起始点$\theta^0=\begin{bmatrix}\theta_1^0\\ \theta_2^0\end{bmatrix}$，计算Loss对它们的偏微分，然后更新$\theta$参数$$\begin{bmatrix}\theta_1^1\\ \theta_2^1\end{bmatrix}=\begin{bmatrix}\theta_1^0\\ \theta_2^0\end{bmatrix}-\eta\begin{bmatrix}\partial L(\theta_1^0)/\partial\theta_1\\ \partial L(\theta_2^0)/\partial\theta_2\end{bmatrix}$$不断重复此步骤。另一种写法：![Pasted image 20240706112434](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706112434.png)
## 调整学习率
1. 学习率太小，梯度下降速度慢（蓝色箭头）；
2. 学习率太大，会使梯度下降速度过大，loss的值会在目标处来回震荡（绿色箭头）。
![Pasted image 20240706113112](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706113112.png)
## 自适应学习率
**方法1**：每轮训练后减少学习率。在训练前期，loss较大时，可以使用较大的学习率使梯度快速下降；在训练达到一定伦次后，loss接近最小值时，减少学习率使其逐步向最小值缓慢收敛。
**方法2**：不同的参数使用不同的学习率。
	Adagrad：将每个参数的学习率除以其先前导数的均方根。原始版本的梯度下降为$$w^{t+1}\leftarrow w^t-\eta^tg^t$$Adagrad更新后的梯度更新方式为$$w^{t+1}\leftarrow w^t-\frac{\eta^t}{\sigma^t}g^t$$其中$\sigma^t$是参数w的先前导数的均值平方根，$\eta^t=\frac{\eta}{\sqrt{t+1}}$，$g^t=\frac{\partial C(\theta^t)}{\partial w}$。![Pasted image 20240706133916](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706133916.png)![Pasted image 20240706134056](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706134056.png)
	梯度更新的式子可以简化为：![Pasted image 20240706135154](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706135154.png)
	![Pasted image 20240706140136](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706140136.png)下方的根号式子可以看作是二次微分

## Stochastic Gradient Descent随机梯度下降
只计算一个x的loss
$$L^n=(\hat{y}^n-(b+\sum w_ix_i^n))^2$$$$\theta^i=\theta^{i-1}-\eta\nabla L^n(\theta^{i-1})$$
### 特征采样-归一化处理
对每一层的input输入特征，进行如下处理:$$x_i^r\leftarrow\frac{x_i^r-m_i}{\sigma_i}$$经过这样处理后，使得每层的均值为0，方差均为1。
# Backpropagation
反向传播可以使得梯度下降计算更有效率

## 链式法则
![Pasted image 20240706142106](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706142106.png)
![Pasted image 20240706142613](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706142613.png)计算forward pass中的$\frac{\partial z}{\partial w}$就是前一层的输入input。
![Pasted image 20240706142845](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706142845.png)在假设了问号处的数值，才能计算$\frac{\partial C}{\partial z}$![Pasted image 20240706142944](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706142944.png)![Pasted image 20240706143057](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706143057.png)一直到网络最后一层，便可以计算之前假设处没有计算的地方