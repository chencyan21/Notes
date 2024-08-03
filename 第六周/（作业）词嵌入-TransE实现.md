# 原理
TransE是用于知识图谱的嵌入表示模型，由Bordes等人于2013年提出。它的主要思想是通过将实体和关系表示为低维向量，使得知识图谱中的三元组（头实体，关系，尾实体）在向量空间中能够以一种简单的几何关系来表示。

具体来说，给定一个三元组$(h, r, t)$，其中$h$是头实体,$r$是关系，$t$是尾实体，TransE的目标是使得头实体向量$h$加上关系向量$r$能够尽量接近尾实体向量 $t$，即：$$h+r\approx t$$例如：
	vec(Rome) + vec(is-capital-of) ≈ vec(Italy)
	vec(Paris) + vec(is-capital-of) ≈ vec(France)
![Pasted image 20240802112925](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240802112925.png)
## 模型细节

1. **嵌入表示**：将实体和关系都嵌入到同一个低维向量空间中。实体和关系的向量分别记作$\mathbf{h}$、$\mathbf{r}$和$\mathbf{t}$。

2. **能量函数**：使用 L1 或 L2 范数来衡量头实体向量加上关系向量与尾实体向量之间的距离。能量函数定义为：$$f(h, r, t) = \|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_1 \quad \text{或} \quad f(h, r, t) = \|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_2$$
3. **训练目标**：通过最小化正确三元组的能量值，同时最大化错误三元组的能量值来训练模型。常用的方法是基于负采样的margin-based ranking loss，定义为：
$$L = \sum_{(h, r, t) \in \mathcal{S}} \sum_{(h', r, t') \in \mathcal{S'}} \left[ \gamma + f(h, r, t) - f(h', r, t') \right]_+$$
其中 $\mathcal{S}$ 是训练集中的正确三元组，$\mathcal{S'}$ 是通过负采样得到的错误三元组，$\gamma$ 是一个超参数（margin），$[\cdot]_+$ 表示取正值。
# 伪代码
![Pasted image 20240802113019](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240802113019.png)
1. 输入input：训练集$S={(h,l,t)}$，实体集E和关系集L，margin值$\gamma$（超参数），嵌入维度k。
2. 初始化：初始化$l$从均匀分布$\text{uniform}(-\frac{6}{\sqrt{k}}, \frac{6}{\sqrt{k}})$采样并对其进行L2 范数归一化处理；初始化$e$从均匀分布$\text{uniform}(-\frac{6}{\sqrt{k}}, \frac{6}{\sqrt{k}})$采样。
3. 循环开始：
	-  对每个$e$进行归一化处理。
4. 取一个小批量的三元组：
	- 从训练集中随机采样一个小批量$S_{\text{batch}}$（大小为b)。
	- 初始化一个空的三元组对集合$T_{\text{batch}}$。
5. 为每个三元组生成负样本：
	- 对于每个小批量中的三元组 $(h, l, t)$，随机生成一个负样本三元组 $(h', l, t')$。
	- 将正负样本对$((h, l, t), (h', l, t'))$添加到集合$T_{\text{batch}}$中。
6. 更新嵌入：
	- 使用梯度下降更新嵌入向量，最小化损失函数：$$\sum_{((h, l, t), (h',l, t')) \in T_{\text{batch}}} \nabla \left[ \gamma + d(h + l, t) - d(h' + l, t') \right]_+$$其中 $d(x, y)$表示向量x和y之间的距离。
7. 循环结束。
# 模型架构分析
## dataloader
dataloader中会获取entity实体，relation关系，triple_list三元组和valid_triple_list验证三元组。同时全局变量`entities2id`和`relations2id`也会进行修改。具体如下：
```
dataloader:
	entities2id:
		key: entity value: id
	relation2id:
		key: relationi value: id
	Returns:
	List[int]: entity列表中包含了id值
	List[int]: relation列表中包含了id值
	List[(int,int,int)]: triple_list三元组中包含了(h,l,t)信息
	List[(int,int,int)]: valid_triple_list三元组中包含了(h,l,t)信息	
```
## 模型E
模型E是计算loss的，有以下方法：
1. `normalization_ent_embedding`：对实体嵌入向量进行L2范数归一化处理
2. `normalization_rel_embedding`：对关系嵌入向量进行L2范数归一化处理
3. `__data_init`：对上述的两个embedding层先进行xavier初始化，然后调用上述两个方法进行L2归一化。
4. `input_pre_transe`：预加载模型参数，如果已经有了训练好的模型参数，可以通过该方法加载参数，而非从随即初始值开始训练。
5. `distance`：计算三者的距离度量`distance = head + rel - tail`，并返回该距离的L1范数。（不同于下面的`test_distance`，该方法好像是私有的，而`test_distance`方法可以被外部访问）
6. `test_distance`：同`distance`但返回的结果在cpu上。
7. `scale_loss`：计算正则化损失，以确保嵌入向量的范数不超过 1。目的是控制嵌入向量的大小，防止过拟合。
8. `forward`：传入的是三元组`current_triples`和负采样得到的错误三元组`corrupted_triples`。首先计算二者的distance，然后将实体、关系分别拼接得到他们的嵌入向量。使用`MarginRankingLoss`计算loss，然后加入正则防止过拟合。
## 模型TransE
模型TransE有以下方法：
1. `data_initialise`：初始化模型E和优化器
2. `insert_pre_data`：读取特定文件并调用模型E中的`input_pre_transe`加载模型参数
3. `insert_test_data`：先调用`insert_pre_data`，然后再读取文件获取`test_triples`测试三元组
4. `insert_traning_data`：先调用`insert_pre_data`，然后再读取文件获取训练时的`train_loss`和`validation_loss`。
5. `training_run`：
	1. 计算`n_batches`和`valid_batch`，分别是train和val的batch次数。
	2. 开始循环`for epoch in range(epochs)`，清零`loss`和`valid_loss`
	3. 训练`for batch in range(n_batches)`，从三元组中随机抽取batch_size个samples
	4. 遍历每个样本，将样本放入到`current`，将负采样的样本放入到`corrupted`
	5. 调用`update_triple_embedding`更新模型参数
	6. 在验证集中`for batch in range(valid_batch)`，与训练类似，最后调用`calculate_valid_loss`得到`valid_loss`
	7. 打印平均的`loss`和`valid_loss`，并存到相关变量中
	8. 训练完成
	9. 使用`plt`绘图
	10. 分别将`ent_embedding`和`rel_embedding`模型参数写入文件，以及loss写入文件
6. `update_triple_embedding`：首先清零优化器梯度，计算loss，然后反向传播计算梯度，最后更新模型参数
7. `calculate_valid_loss`：计算`valid_loss`（累积的）
8. `test_run`：
	1. 遍历测试三元组`for triple in self.test_triples`，定义相关变量
	2. 遍历所有实体entity，替换head_entity，计算距离
	3. 遍历所有实体entity，替换tail_entity，计算距离
	4. 使用`sorted`按照`distance`的大小对三元组进行排序
	5. 比对`head`和`tail`的结果，如果在前十那么`hits`加一，`rank_sum`累加
	6. 计算`hit@10`和`meanrank`。
# 模型调优
`batch_size`定为最大，最大不能超过`valid_triples`的大小，所以`batch_size=50000`。
为选择`margin`的最佳值，构建列表`margins=[1,2,4,5,10]`依次尝试，结果比对后`margin`为2时效果最佳。
由于增大了batchsize的值，模型的泛化能力会下降许多。相比于原始的结果，增大了batchsize值的结果表现会差一些。
```
load file...
Complete load. entity : 14951 , relation : 1345 , train triple : 483142, , valid triple : 50000
margin:  1
batch size:  9 valid_batch:  2
100%|██████████| 500/500 [53:48<00:00,  6.46s/it] epoch:  499 cost time: 6.824
Train loss:  tensor(0.0048, device='cuda:0', grad_fn=<DivBackward0>) Valid loss:  tensor(0.0862, device='cuda:0', grad_fn=<DivBackward0>)

100%|██████████| 59071/59071 [45:40<00:00, 21.56it/s]  
hits@10:  0.4240067037971255
meanrank:  288.3658309491967
margin:  2
batch size:  9 valid_batch:  2
100%|██████████| 500/500 [53:57<00:00,  6.48s/it]epoch:  499 cost time: 6.86
Train loss:  tensor(0.0170, device='cuda:0', grad_fn=<DivBackward0>) Valid loss:  tensor(0.2059, device='cuda:0', grad_fn=<DivBackward0>)

100%|██████████| 59071/59071 [44:46<00:00, 21.99it/s]  
hits@10:  0.45693318210289313
meanrank:  247.69046571075486
margin:  4
batch size:  9 valid_batch:  2
100%|██████████| 500/500 [56:11<00:00,  6.74s/it]epoch:  499 cost time: 6.717
Train loss:  tensor(0.1156, device='cuda:0', grad_fn=<DivBackward0>) Valid loss:  tensor(0.6143, device='cuda:0', grad_fn=<DivBackward0>)

100%|██████████| 59071/59071 [46:14<00:00, 21.29it/s]  
hits@10:  0.40451321291327386
meanrank:  265.2810600802424
margin:  5
batch size:  9 valid_batch:  2
100%|██████████| 500/500 [56:14<00:00,  6.75s/it] epoch:  499 cost time: 6.647
Train loss:  tensor(0.2324, device='cuda:0', grad_fn=<DivBackward0>) Valid loss:  tensor(0.9034, device='cuda:0', grad_fn=<DivBackward0>)

100%|██████████| 59071/59071 [46:43<00:00, 21.07it/s]  hits@10:  0.3649675813851128
meanrank:  316.00402058539726
```


