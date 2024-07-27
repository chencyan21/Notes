# 迁移学习
自监督学习：使用嵌入在自变量中的标签来训练模型，而不需要外部标签。例如，训练一个模型来预测文本中的下一个单词。

通用语言模型微调（ULMFit）：在将学习转移到分类任务之前，对语言模型进行额外阶段的微调，可以显著提高预测效果。有以下三个阶段：1.引入预训练模型，2.在预训练模型上进行微调，3.处理分类任务。
![Pasted image 20240724220617](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240724220617.png)
# 文本预处理
将数据集中所有文档连接成一个大的长字符串，然后将其拆分为单词，得到一个很长的单词列表/tokens。自变量为第一个单词到倒数第二单词的序列，因变量为第二个单词到最后一个单词的序列。

Vocab（词汇表）
* **定义**：词汇表是一个包含特定语言或文本数据集中所有独特单词的集合。通常，它是一个有限的列表，用于表示可以在文本中出现的所有单词。
* **用途**：词汇表用于文本处理和特征提取。例如，在构建词向量（如Word2Vec或GloVe）时，需要一个词汇表来确定哪些单词需要向量表示。在语言模型中，词汇表用于限定模型可以预测的单词范围。
* **特点**：词汇表的大小通常较小，因为它只包含独特的单词。
* **例子**：“猫在桌子上睡觉”，词汇表可能是：`["猫", "在", "桌子", "上", "睡觉"]`。
Corpus（语料库）：
* **定义**：语料库是一个包含大量文本数据的集合，通常用于训练和评估NLP模型。
* **用途**：语料库用于训练语言模型、构建词向量、进行文本分析等任务。它提供了丰富的上下文信息，有助于模型学习语言模式和语义关系。
* **特点**：语料库通常非常大，包含大量句子和单词，以便提供足够的数据来训练复杂的模型。
* **例子**：语料库可能包含许多句子，例如：1. 猫在桌子上睡觉。2. 狗在院子里玩耍。3. 天气很好。

创建语言模型所需具体步骤：
1. **Tokenization标记化**：将文本转换为单词列表或者字符。
2. **Numericalization数值化**：列出所有出现的独特单词（构建词汇表），并通过查找其在词汇表中的索引，将每个单词转换为数字。
3. **Language model data loader creation语言模型数据加载器的创建**：构建batch。fastai 提供了一个LMDataLoader类，可自动处理因变量的创建，该因变量与自变量偏移一个token。它还处理了一些重要细节，例如如何以因变量和自变量按要求保持结构的方式洗牌训练数据。
4. **Language model creation创建语言模型**：需要一种特殊的模型，例如RNN，它能做处理可任意增大或减小的输入列表。
## Tokenization
针对不同的语言特性，tokenize的方法主要有三种：
1. 基于单词（word-based）：根据空格分隔句子，即使没有空格，也会应用特定的语言规则来分隔意义部分（例如将 "don't "变成 "do n't"）。标点符号也会被分割成单独的标记。例如英文。
2. 基于子词（subword-based）：根据最常出现的子串将单词分割成更小的部分。例如，"场合 "可以标记为 "o c ca sion"。例如中文。
3. 基于字符（character-based）：将句子分割成单个字符。
### 使用fastai进行基于单词的tokenize
fastai提供了API`WordTokenizer()`可将句子tokenize为单词
```Python
spacy = WordTokenizer()
first(spacy(['The U.S. dollar $1 is $1.00.']))
# output:
# (#9) ['The','U.S.','dollar','$','1','is','$','1.00','.']
```

spacy可以很好处理细节，能够将“it's”分割成“it”和“'s”，并且在缩写词和数字中处理得当（上面的输出）。

fastai还会通过`Tokenizer`类为tokenization过程添加一些额外的功能：
```Python
tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 31))
# output:
# (#207) ['xxbos','xxmaj','once','again','xxmaj','mr','.','xxmaj','costner','has','dragged','out','a','movie','for','far','longer','than','necessary','.','xxmaj','aside','from','the','terrific','sea','rescue','sequences',',','of','which'...]
```
**其中的`xx`在英文中并不常见，所以用来表示特殊标记**。例如，`xxbos`表示新文本的开始（beginning of sequence/stream），通过这个起始标记，模型能够学习到：忘记前面学习的内容，并开始学习后面的词语。
fastai使用的特殊标记：
* `xxbos`：表示文本的开头
* `xxmaj`：表示下一个单词以大写字母开头 
- `xxunk`：表示单词未知
### 使用fastai进行基于子词的tokenize
子词的tokenize步骤：
1. 分析corpus，找出最常出现的字母组合，这些将构建为vocab；
2. 使用vocab中的子词单元对corpus进行tokenize。

如果vocab中的子词数量越小，那么每个token表示的字符数就会越少，当达到最小时，就是字符tokenize。相反，如果vocab的子词数量越大，那么每个token就可以涵盖越多的单词，直到涵盖所有的英文单词，此时就是单词tokenize。
## Numericalization
Numericalization数值化是将token映射为整数的过程。在tokenize过程完成后，使用`setup`函数构建vocab。

```Python
num = Numericalize()
num.setup(toks200)
coll_repr(num.vocab,20)
nums = num(toks)[:20]
' '.join(num.vocab[o] for o in nums)
```
代码段中num便是Numericalize对象，将文本信息分割成token后传入num便得到数值化后的结果nums，通过num.vocab将数值化的nums重新转化为tokens。
## 将text以batch形式传入模型
希望语言模型能够按照顺序阅读文本，也就意味着，每一个batch的文本都应该从上一个batch结束的地方开始。
```Python
nums200 = toks200.map(num)
```
类比于tokenize的过程，在map方法中传入Numericalize对象num便可以对toks200进行Numericalization。
通过`LMDataLoader`自动将`nums200`分割成batch个`batch_size*seq_len`，默认`seq_len`为72，`batch_size`为64。
```Python
dl = LMDataLoader(nums200)
total_lens=0
for i in nums200:
    total_lens+=len(i)
print(total_lens)# 56228
len(dl)# 13
x,y = first(dl)
x.shape,y.shape# (torch.Size([64, 72]), torch.Size([64, 72]))
```
实际分割下的结果为：`batch(13)*batch_size(64)*seq_len(72)=59904`，考虑到最后一个batch不能划分完全，所以结果是合理的。
# Training a Text Classifier训练文本分类器/模型
通过上述的步骤，得到了dataloader出来的结果x和y，其中x和y的结果相差一个字符，y比x要慢一步。
## Fine-Tuning
使用embedding将整数词索引转化为神经网络使用的激活度，将词汇从高维的离散空间（如Vocab的索引）映射到低维的连续向量空间。
使用AWD-LSTM架构，损失函数为CrossEntropyLoss。
```Python
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3,
    metrics=[accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(1, 2e-2)
```
`fit_one_cycle`方法可以动态调整学习率，在训练的前半部分，逐步增加学习率。在训练的后半部分，逐步降低学习率。参数为`epoches=1`训练总轮数，`2e-2`为最大学习率。
模型`language_model_learner` 会自动调用 `freeze` ，因此只会训练全连接层。
> 在`language_model_learner`中，`get_language_model`函数将预训练模型`AWD_LSTM`作为encoder，以及`LinearDecoder`作为decoder，合为一个模型后返回。
> ![Pasted image 20240726174629](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240726174629.png)
> ![Pasted image 20240726174754](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240726174754.png)
## 模型保存与加载
模型保存：
```Python
learn.save('1epoch')
```
会在`learn.path/models/`下创建一个`1epoch.pth`的文件。
模型加载：
```Python
learn = learn.load('1epoch')
```
初步训练完成后，在解冻后对模型进行微调。
```Python
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)  
learn.save_encoder('finetuned')
```
训练完成后，保存该模型的全部内容，除了最后一层，最后一层将会更改为分类器。

先训练模型头部而不是直接训练整个模型主要是为了充分利用预训练模型的特征，同时避免训练过程中的不稳定性和计算资源浪费。
1. **充分利用预训练模型的特征**。预训练模型（例如 ResNet、VGG 等）通常是在大规模数据集（如 ImageNet）上训练得到的。这些模型已经学习到了很多通用的低级和中级特征（如边缘、纹理、形状等），这些特征在各种视觉任务中都是有用的。
- **冻结预训练层**：在训练初期，通过冻结预训练模型的卷积层（特征提取部分），我们可以充分利用这些已经学到的特征。
- **训练新的分类头部**：只训练新添加的分类头部（全连接层），以适应当前特定任务的数据分布和类别。
2. **稳定训练过程**。直接训练整个模型可能导致训练过程不稳定，尤其是在新任务的数据量较少或者数据分布与预训练数据有较大差异时。
- **减少不稳定性**：通过先训练头部，可以逐步调整模型的参数，避免训练过程中出现大幅度的权重更新，导致模型不稳定。
- **逐步调整学习率**：先训练头部可以帮助找到合适的学习率和训练参数，解冻后微调整个模型时可以更平滑地进行权重更新。
3. **节省计算资源**。直接训练整个模型需要更多的计算资源和时间，特别是当预训练模型较大时（如 ResNet-152、VGG-19 等）。
- **减少计算量**：先冻结大部分层，只训练头部，可以大大减少计算量，加速初期的训练过程。
- **高效利用资源**：在模型头部训练达到一定效果后，再解冻整个模型进行微调，可以更高效地利用计算资源，提高训练速度。
## Text Generation文本生成/微调分类器前模型预测
在微调分类器前，模型可以输出一些评论。由于训练的模型可以猜测句子的下一个单词是什么，因此可以让模型撰写新的评论：
```Python
TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75)
         for _ in range(N_SENTENCES)]
# output:
# i liked this movie because it was a " b " movie . If you like some of the Hollywood " hollywood " approach to film - making , then you will hate this . If you liked Cabin Fever
# i liked this movie because it taught me a lot about love and a happy life . i liked it at the same time as it was one of my favourites . a classic for all ages . The only thing that please make
```
### 创建分类器dataloader
模型需要从语言模型微调转为分类器微调。语言模型预测文档的下一个单词，因此它不需要任何外部标签。而分类器则会预测一些外部标签。
在IMDB数据集，分为了`pos`和`neg`的文件夹，可以构建特征-标签对。在创建`dataloader`时，使用的`vocab`必须时语言模型微调的`vocab`。
```Python
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()
learn = learn.load_encoder('finetuned')
```
预训练模型仍使用`AWD_LSTM`，并且使用`finetued`模型参数文件加载encoder参数。
**注**：加载参数中没有分类器的参数，因为前面保存参数时只保存了encoder部分参数（倒数第二层及以前）
> 在构建模型中，与语言模型构建类似，`text_classifier_learner`函数中的`get_text_classifier`函数构建了将encoder和decoder相结合。其中的decoder部分是PoolingLinearClassifier。
> ![Pasted image 20240726201010](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240726201010.png)![Pasted image 20240726201040](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240726201040.png)

## 微调分类器
与微调语言模型类似，都是先解冻再训练。
```Python
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
```
# 总结
模型微调的具体步骤：
- **准备数据/文本预处理**
- **加载预训练模型和dataloader**
- **冻结预训练层**
- **训练模型头部**
	- 先训练头部而不是直接训练整个模型的原因：**利用预训练特征**：充分利用预训练模型中已经学到的通用特征。**稳定训练过程**：减少训练过程中权重更新的不稳定性。**节省计算资源**：减少计算量，提高训练效率。
- **解冻所有层**
- **微调整个模型**
- **评估和保存模型**
# Questionnaire
1. What is "self-supervised learning"?什么是 "自监督学习"？
	1. 使用嵌入在自变量中的标签来训练模型，而不需要外部标签。例如，训练一个模型来预测文本中的下一个单词。
2. What is a "language model"?什么是 "语言模型"？
	1. 一种统计模型或计算模型，用于估计一个词序列的概率。
3. Why is a language model considered self-supervised?为什么语言模型被认为是自监督的？
	1. 语言模型在训练时使用的是大量未经标注的文本。
4. What are self-supervised models usually used for?自监督模型通常用于哪些方面？
	1. 应用于NLP和cv。
5. Why do we fine-tune language models?
	1. 预训练模型通常在大规模的数据集上训练，能力更好，通用性好，但是为了使模型更适应当前的任务，需要使用微调让模型更好适应这些特定需求。
6. What are the three steps to create a state-of-the-art text classifier?创建最先进文本分类器的三个步骤是什么？
	1. 数据准备、模型选择与训练、模型评估与优化。
7. How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?50,000 篇未标记的电影评论如何帮助我们为 IMDb 数据集创建更好的文本分类器？
	1. 用于微调预训练模型，让预训练模型更好地适应当前的影评任务。
8. What are the three steps to prepare your data for a language model?为语言模型准备数据的三个步骤是什么？
	1. tokenize、numericalize和将数据装入dataloader
9. What is "tokenization"? Why do we need it?什么是 "标记化"？我们为什么需要它？
10. Name three different approaches to tokenization.说出标记化的三种不同方法。
11. What is `xxbos`?什么是 `xxbos` ?
12. List four rules that fastai applies to text during tokenization.列出 fastai 在标记化过程中应用于文本的四条规则。
13. Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?为什么重复字符会被替换为显示重复次数和重复字符的标记？
14. What is "numericalization"?什么是 "数值化"？
	1. 将文本转换为单词列表（或字符或子串，具体取决于模型的粒度）
15. Why might there be words that are replaced with the "unknown word" token?为什么会出现用 "未知单词 "标记替换的单词？
	1. 有一些特殊符号，例如emoji等。
16. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)批量大小为 64，代表第一批的张量的第一行包含数据集的前 64 个标记。该张量的第二行包含什么？第二批的第一行包含什么？(注意--学生们经常会弄错这个问题！请务必在本书的网站上查看您的答案）
	1. 第一批第二行：包含前64个样本在第二个序列位置的标记，第二批第一行：包含第65到128个样本在第一个序列位置的标记。
17. Why do we need padding for text classification? Why don't we need it for language modeling?为什么文本分类需要填充？为什么语言建模不需要？
	1. 文本分类中填充的目的使为了确保所有的输入序列长度保持一致，可以以一个batch传入。语言模型中能够处理变长序列，或者通过掩码机制来忽略。
18. What does an embedding matrix for NLP contain? What is its shape?NLP 的嵌入矩阵包含哪些内容？它的形状是什么？
	1. 应该包含vocab中的所有词汇的向量表示。形状是(vocab大小，embedding的维度)
19. What is "perplexity"?什么是 "困惑度"？
	1. 困惑度是语言模型对一个给定文本序列的“困惑”程度的度量。直观地说，它反映了模型对预测下一个词的难度。困惑度越低，表示模型越能准确地预测下一个词，性能越好。
20. Why do we have to pass the vocabulary of the language model to the classifier data block?为什么要将语言模型的词汇传递给分类器数据块？
	1. 确保使用的token与index关系保持一致。
21. What is "gradual unfreezing"?什么是 "逐步解冻"？
	1. 先冻结预训练模型的所有层，然后训练新添加的层（全连接层），训练一段时间后，逐步解冻预训练模型的部分层，从高到低，逐渐向下解冻，可以让模型在训练过程中逐渐调整深层特征。
22. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?为什么文本生成总是可能领先于机器生成文本的自动识别？
	1. 文本生成技术的迅速发展和复杂性，往往使得生成的文本质量高到难以区分与人工创作的区别。