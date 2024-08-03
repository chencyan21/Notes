# 模型介绍
OpenNRE是一个用于关系抽取的开源工具包。关系抽取是自然语言处理中的一个关键任务，旨在从文本中识别并提取实体之间的语义关系。OpenNRE 由KEG团队开发，支持多种关系抽取模型，并提供了简单易用的接口，方便研究人员和开发者进行关系抽取任务。
## 主要模块
1. 数据模块：
   OpenNRE 支持多种数据格式，提供了数据预处理工具，包括实体标注、关系标注等。
2. 模型模块：
   OpenNRE 内置了多种关系抽取模型，如 CNN、RNN、BERT 等，用户可以通过简单配置选择使用。
3. 训练和评估模块：
   OpenNRE 提供了便捷的训练和评估接口，用户可以快速训练模型并进行性能评估。
# 模型测试
使用测试案例：
```Python
model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
```
结果输出：
![Pasted image 20240803144507](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240803144507.png)

## 训练特定数据集
使用如下命令行
```bash
python example/train_supervised_cnn.py \
    --metric acc \
    --dataset semeval \
    --batch_size 160 \
    --lr 0.1 \
    --weight_decay 1e-5 \
    --max_epoch 100 \
    --max_length 128 \
    --seed 42 \
    --encoder pcnn 
```
使用cnn关系抽取模型训练wiki20m数据集，运行结果如下：
![Pasted image 20240803150004](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240803150004.png)