# Glove模型
GloVe是一种用于自然语言处理的词向量表示模型。GloVe 模型通过将词嵌入表示成实数值向量来捕捉词与词之间的语义关系。与基于局部上下文的词向量模型（如 Word2Vec）不同，GloVe 模型是基于全局共现矩阵的。
GloVe 模型的核心思想是通过统计词在大规模语料库中共现的频率来学习词的向量表示。它使用共现矩阵，其中每个元素表示词对在一定窗口大小内共现的次数。GloVe 模型将这种共现信息转化为一个优化问题，通过最小化损失函数来学习词向量。
## 具体步骤
1. 构建共现矩阵：
	1. 对给定corpus，统计词与词之间的共现频率，构建一个共现矩阵$X$，其中$X_{ij}$表示词$i$和词$j$在给定窗口大小内共现的次数。
2. 构造目标函数：
	1. GloVe模型的目标是找到词向量，使得词对的共现概率能够被这些词向量的点积很好地表示。损失函数为：$$J = \sum_{i,j=1}^V f(X_{ij}) (\mathbf{w}_i^T \mathbf{w}_j + b_i + b_j - \log X_{ij})^2$$ $\mathbf{w}_i$和$\mathbf{w}_j$是词i和j的词向量，$b_i$和$b_j$是偏置项，$f(X_{ij})$是一个加权函数，用于平衡频繁和不频繁共现词对的影响。
3. 加权函数：加权函数$f(X_{ij})$设计为：$$f(X_{ij}) = \left\{\begin{array}{ll}(X_{ij} / X_{\max})^\alpha & \text{if } X_{ij} < X_{\max} \\1 & \text{otherwise}\end{array}\right.$$其中，$\alpha$通常取 0.75，$X_{\max}$是一个超参数，用于控制权重的平滑程度。
4. 优化过程：
	1. 使用随机梯度下降法或其他优化算法最小化损失函数，得到词向量$\mathbf{w}_i$。
# 代码使用
运行`./demo.sh`即可运行，首先下载样例数据集`text8.zip`，
```bash
if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi
```
接着从corpus中生成vocab，计算词对的共现矩阵，对共现矩阵进行打乱，然后训练 GloVe 模型并评估结果。
1. 生成词汇表：
	1. 调用`vocab_count`工具从corpus中生成vocab，并将结果写入词汇文件 `$VOCAB_FILE`。
```bash
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
```
2. 生成共现矩阵：
	1. 调用`cooccur` 工具，用于生成词对的共现矩阵，并将结果写入共现文件 `$COOCCURRENCE_FILE`。
```bash
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
```
3. 打乱共现矩阵：
	1. `shuffle` 工具用于将共现矩阵文件打乱，以便训练过程中更好地随机化数据，并将结果写入打乱后的文件 `$COOCCURRENCE_SHUF_FILE`。
```Bash
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
```
4. 训练 GloVe 模型：
	1. 使用`glove` 工具用于训练 GloVe 模型。
```bash
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
```
5. 评估模型：
	1. 如果语料库是 `text8`，则运行评估脚本。
	2. 根据第一个参数 `$1` 的值，选择不同的评估工具：`matlab`、`octave` 或 `python`。
	3. 评估脚本 `read_and_evaluate.m` 和 `read_and_evaluate_octave.m` 分别是 Matlab 和 Octave 版本，`evaluate.py` 是 Python 版本。
```Bash
if [ "$CORPUS" = 'text8' ]; then
   if [ "$1" = 'matlab' ]; then
       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2
   elif [ "$1" = 'octave' ]; then
       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
   else
       echo "$ $PYTHON eval/python/evaluate.py"
       $PYTHON eval/python/evaluate.py
   fi
fi
```
## 运行结果
`vocab_count`工具:  用于计算原文本的单词统计（生成`vocab.txt`,每一行为：单词 词频）
`cooccur`工具：统计词与词的共现，类似word2vec的窗口内的任意两个词（生成`cooccurrence.bin`,二进制文件，太大了没有上传）
`shuffle`：对于cooccur中的共现结果重新整理，即word2vec的窗口内的任意两个词（生成 `cooccurrence.shuf.bin`,二进制文件，太大了没有上传）
`glove`：​​​​​​​glove算法的训练模型，会运用到之前生成的相关文件，最终会输出`vectors.txt`和`vectors.bin`

