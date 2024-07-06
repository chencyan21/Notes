# CNN
## Image Classification
一张图片是3维的tensor，分别代表图片宽、高、通道数channel。
图像如果使用全连接层处理，参数会大大增加，容易造成过拟合现象。

关注图片中的局部信息，例如图中的鸟嘴、鸟眼等局部信息来判断是否是鸟。
![Pasted image 20240706143827](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706143827.png)
## 卷积Convolution
每个神经元只关注部分的信息，称为感受野（receptive field），彼此之间的感受野是可以重叠的。不同的神经元可以有不同大小的感受野，也可以只覆盖某个通道（rgb中只关注red），也可以是长方形。![Pasted image 20240706144112](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706144112.png)
**简化例子1**：涉及所有通道，kernel大小设定为3，stride为2，超出的部分使用padding补充，![Pasted image 20240706144601](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706144601.png)**简化例子2**：神经元之间共享参数，因为输入不同，所以输出会不同。
![Pasted image 20240706144859](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706144859.png)
## Pooling池化层
pooling将图片变小，用来减少计算量。pooling分为max-pooling和average-pooling。以max-pooling为例：
![Pasted image 20240706145752](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706145752.png)
## 整体架构
![Pasted image 20240706145642](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706145642.png)图片通过“卷积-池化”操作不断提取特征，最后使用flatten展开，使用softmax进行分类。
## CNN应用
![Pasted image 20240706145940](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706145940.png)将棋盘中落子 看作分类问题，棋盘上每个可以落子的位置都是一个选项。
# Self-Attention
## 注意力机制
在注意力机制中，将自主性提示称为查询（query），查询指的是对所有信息中的信息进行查询得到需要的信息，忽略其余的不相关信息。给定任何查询，注意力机制通过注意力汇聚层（attention pooling）将输入的查询引导至感官输入。在注意力机制中，这些感官输入被称为值（value），每个值都与一个键（key）配对。
## QKV计算
![Pasted image 20240706152031](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706152031.png)
![Pasted image 20240706152325](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706152325.png)
只有$W^q$、$W^k$、$W^v$是未知的，需要学习的，其他的参数都是已知的。
## 多头自注意力
如果相关信息有多种，那么可以设置多个head来学习多种相关信息。
![Pasted image 20240706153032](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706153032.png)
为每一个位置设置一个向量positional encoding，可以包含位置信息，使得模型能够区分不同位置的词语。
## 与其他模型比对
### 与CNN比对
CNN只考虑感受野中的信息，而self-attention考虑的是整个图的信息，因此可以将CNN看作是简化版本的自注意力。反过来说，Self-attention是复杂化的CNN。![Pasted image 20240706153332](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706153332.png)随着数据集越大，Self-attention的效果比CNN更好。![Pasted image 20240706153515](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706153515.png)
## 与RNN比对
Self-attention考虑了双向的信息，而单向RNN不能看到右侧的信息，双向RNN才能看到。
RNN生成的结果是有先后顺序的，不能像Self-attention可以平行处理所有数据。
![Pasted image 20240706153845](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706153845.png)
## Self-attention for Graph
self-attention可以看作是一种GNN，只需要计算相连节点的值。![Pasted image 20240706154134](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240706154134.png)