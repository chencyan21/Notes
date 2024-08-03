# 数据准备
## Tokenization
英文直接分隔字符，中文使用jieba分词，然后在句子的前后添加`BOS`和`EOS`，以这样的方式处理训练集和验证集，得到`train_en`、`train_cn`、`dev_en`和`dev_cn`。
## Numericalization
使用`Counter`统计所有的单词数量，并构建vocab/dict。构建完成前还要添加`UNK`和`PAD`两个特殊的符号。
## 句子padding
由于句子长短不一，先获取最长的句子长度，然后将其他句子扩张到相同长度，用0填充。
```Python
def sent_padding(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths) # 取出最长的的语句长度
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths
```
# 模型架构
## Encoder部分
`forward`函数的输入是x和其长度，按输入序列的长度进行排序，将x通过embedding层后，使用dropout丢弃部分数据，通过rnn得到输出和隐状态，将双向GRU的最后一层的正向和反向隐藏状态拼接，并通过fc层和激活函数处理。最后返回的是编码器输出和隐状态。
## Decoder部分
`decoder`中`create_mask`方法用于创建一个掩码，用于在计算注意力时忽略输入的某些部分。在`forward`函数中，对`y_lengths`进行排序，并相应调整`y`和`hid`，对排序后的`y`进行embedding使用dropout，通过RNN，然后将输出解包，恢复序列的原始顺序。调用`create_mask`方法，在计算注意力时忽略填充标记。使用attention将编码器输出和解码器的隐状态结合，对最终输出应用 log softmax。
## Model: Seq2seq
将encoder和decoder两个部分结合起来，并且创建一个方法`translate`用于预测结果。
# 训练
## 参数选择
`dropout=0.2`，隐层维度为100，训练轮数为30。
由于数据集数据量较小，所以结果并不算好。
## Loss
`LanguageModelCriterion` 类定义了一种计算交叉熵损失的标准，用于语言模型中的掩码（masked）交叉熵损失。
```Python
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
    def forward(self, input, target, mask):
        input = input.contiguous().view(-1, input.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output
```
