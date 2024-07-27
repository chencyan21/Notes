# 使用Mid-Level API进行数据处理
mid-level API不仅包含创建`dataloader`的功能，还具有回调系统和通用优化器功能。
## 分层API
### Transform类
在上一节中的tokenization和numericalization中，有一个相同的`decode`方法。
```Python
nums_dec = num.decode(nums[0][:10])
# output:
# (#10) ['xxbos','xxmaj','once','again','xxmaj','mr','.','xxmaj','costner','has']
tok.decode(nums_dec)
# output:
# 'xxbos xxmaj once again xxmaj mr . xxmaj costner has'
```
`Numericalize.decode`将索引逆向为字符串标记/分词，`Tokenizer.decode`将分词返回为单个字符串（由于添加了特殊字符等，结果与原始字符串有些区别）。

对于前面示例中的每个 `tok` 或 `num` ，创建了一个名为 `setup` 方法的对象，将其应用于原始文本，然后最终将结果解码回可理解的表示形式。大多数数据预处理任务都需要这些步骤，因此 fastai 提供了一个类来封装它们。这是 `Transform` 类。 `Tokenize` 和 `Numericalize` 都是**继承**自 `Transform` 。

一般来说， `Transform` 是一个行为类似于函数的对象，并且有一个可选的 `setup` 方法，该方法将初始化一些内部状态（如 `num` 中的词汇）以及一个可选的 `decode` ，它将反转该函数。

编写自定义的 `Transform`有两种方法：
1. 编写一个函数，然后使用 `Transform`来匹配：
```Python
def f(x:int): return x+1
tfm = Transform(f)
```
2. 使用装饰器：
```Python
@Transform
def f(x:int): return x+1
f(2),f(2.0)
```
如果需要`setup`等方法，则需要继承父类`Transform`来实现：
```Python
class NormalizeMean(Transform):
    def setups(self, items): self.mean = sum(items)/len(items)
    def encodes(self, x): return x-self.mean
    def decodes(self, x): return x+self.mean
```
**注意**：具体的实现和调用是不同的，例如，实现的是`setups`，调用的方法是`setup`。

| Class                 | To call                       | To implement |
| --------------------- | ----------------------------- | ------------ |
| `nn.Module` (PyTorch) | `()` (i.e., call as function) | `forward`    |
| `Transform`           | `()`                          | `encodes`    |
| `Transform`           | `decode()`                    | `decodes`    |
| `Transform`           | `setup()`                     | `setups`     |
### Pipeline类
为了将多个`transforms`组合在一起，可以使用`pipeline`。通过向`transforms`的列表传入`pipeline`，该类实例将会在其中按顺序进行组合`transforms`。
```Python
tfms = Pipeline([tok, num])
t = tfms(txts[0]); t[:20]
```
相对应地，使用`decode`方法也会按照相反顺序依次调用`decode`。
## TfmdLists and Datasets: Transformed Collections
在fastai中，一系列`transforms`由`Pipeline`表示。将此 `Pipeline` 与原始项目组合在一起的类称为 `TfmdLists` 。
### TfmdLists类
初始化时，`TfmdLists`会按照顺序调用每个`Transform`的`setup`方法。对应地，它也有`decode`方法。
```Python
tls = TfmdLists(files, [Tokenizer.from_folder(path), Numericalize])
tls.decode(t)[:100]
```
此外，它还可以使用`splits`参数来分隔train和valid，通过`train`和`vaild`属性可以进行访问。
```Python
cut = int(len(files)*0.8)
splits = [list(range(cut)), list(range(cut,len(files)))]
tls = TfmdLists(files, [Tokenizer.from_folder(path), Numericalize], splits=splits)  
tls.valid[0][:20]
```
#### 创建并行`transforms`前提-创建标签
一般来说，需要有两个（或更多）并行的`pipelines`：一个用于将原始项目处理为输入，另一个用于将原始项目处理为目标。
首先，利用函数 `parent_label`从父文件夹中获取标签名称。然后需要一个 `Transform` 叫做 `Categorize`来获取唯一的项目并在`setup`过程中用它们构建一个`vocab`，然后在调用时将字符串标签转换为整数。
```Python
lbls = files.map(parent_label)
cat = Categorize()
cat.setup(lbls)
tls_y = TfmdLists(files, [parent_label, Categorize()])
```
### Datasets类
`Datasets` 将并行应用两个（或更多）`pipeline`，结果返回一个元组。当索引到 `Datasets` 时，它将返回一个包含每个`pipeline`结果的元组：
```Python
x_tfms = [Tokenizer.from_folder(path), Numericalize]
y_tfms = [parent_label, Categorize()]
dsets = Datasets(files, [x_tfms, y_tfms])
x,y = dsets[0]
```
同样也可以在`Datasets`中添加`splits`分隔数据集：
```Python
x_tfms = [Tokenizer.from_folder(path), Numericalize]
y_tfms = [parent_label, Categorize()]
dsets = Datasets(files, [x_tfms, y_tfms], splits=splits)
x,y = dsets.valid[0]
```
也可以对`dsets`使用`decode`方法。
最后需要使用`dataloaders`方法将`Datasets`转为`DataLoaders`，传入参数`before_batch=pad_input`填充空缺。
```Python
dls = dsets.dataloaders(bs=64, before_batch=pad_input)
```
## Mid-Level API应用
当在 `TfmdLists` 或 `Datasets` 对象上调用 `show` 方法时，它会对项目进行解码，直到解码到一个包含 `show` 方法的类型，然后使用它来显示对象。该 `show` 方法会传递一个 `ctx` ，对于图像，它可以是一个 `matplotlib` 轴，对于文本，它可以是 DataFrame 的一行。
```Python
class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        img1,img2,same_breed = self
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2),
                          title=same_breed, ctx=ctx)
img = PILImage.create(files[0])
s = SiameseImage(img, img, True)
s.show();
```
# Questionnaire
1. Why do we say that fastai has a "layered" API? What does it mean?为什么说 fastai 有一个 "分层 "的应用程序接口？这是什么意思？
	1. 高层的API接口类似于10NLP中使用的接口tokenizer等，中层的API接口类似于本章的transforms。高层的API中各种函数方法都已经封装好了，可以直接使用，中层的API适用于自定义一些transforms。
2. Why does a `Transform` have a `decode` method? What does it do?为什么 `Transform` 有 `decode` 方法？它有什么作用？
	1. 可以用于数据反向变换，逆向操作，可以对比数据前后的变化。
3. Why does a `Transform` have a `setup` method? What does it do?为什么 `Transform` 有 `setup` 方法？它有什么作用？
	1. 在数据变化之前进行一些准备工作，例如计算标准差。
4. How does a `Transform` work when called on a tuple?在元组上调用 `Transform` 时如何工作？
	1. 可以直接用于元组tuple。
5. Which methods do you need to implement when writing your own `Transform`?在编写自己的 `Transform` 时，您需要实现哪些方法？
	1. `encodes`（调用的是`()`）、`decodes`（调用的时候是`decode()`）和`setups`（调用的时候是`setup()`）
6. Write a `Normalize` transform that fully normalizes items (subtract the mean and divide by the standard deviation of the dataset), and that can decode that behavior. Try not to peek!编写一个 `Normalize` 转换，将项目完全标准化（减去平均值并除以数据集的标准偏差），并能对该行为进行解码。尽量不要偷看！
```Python
class NormalizeTransform(Transform):
    def setups(self, items): 
	    self.mean = sum(items)/len(items)
    def encodes(self, x): 
	    return x-self.mean
    def decodes(self, x): 
	    return x+self.mean
```
7. Write a `Transform` that does the numericalization of tokenized texts (it should set its vocab automatically from the dataset seen and have a `decode` method). Look at the source code of fastai if you need help.编写一个 `Transform` ，对标记化文本进行数值化处理（它应根据所看到的数据集自动设置词汇表，并有一个 `decode` 方法）。如果需要帮助，请查看 fastai 的源代码。
```Python
class CustomTransform(Transform):
    def __init__(self, vocab=None, min_freq=3, max_vocab=60000, special_toks=None):
        self.vocab=vocab
        self.min_freq=min_freq
        self.max_vocab=max_vocab
        self.special_toks=special_toks
        self.o2i=None
    def setups(self,dsets):
        self.dsets=dsets
        count=Counter(p for o in self.dsets for p in o)
        self.vocab = make_vocab(count, min_freq=self.min_freq, max_vocab=self.max_vocab,special_toks=self.special_toks)
        self.o2i={i:j for i,j in enumerate(self.vocab)}
    def encodes(self,o):
        return TensorText(tensor([self.o2i[i] for i in o]))
    def decodes(self, o):
        return L(self.vocab[o_] for o_ in o)
```
8. What is a `Pipeline`?什么是 `Pipeline` ?
	1. 多个transform的集合。
9. What is a `TfmdLists`?什么是 `TfmdLists` ?
	1. 将pipeline和原始的数据文件合并在一起。
10. What is a `Datasets`? How is it different from a `TfmdLists`?什么是 `Datasets` ？它与 `TfmdLists` 有什么不同？
	1. 相比于`TfmdLists` ，它可以应用并行`pipeline`。
11. Why are `TfmdLists` and `Datasets` named with an "s"?为什么 `TfmdLists` 和 `Datasets` 以 "s "命名？
	1. 表示的是一组数据的集合，而不是单一的数据项。
12. How can you build a `DataLoaders` from a `TfmdLists` or a `Datasets`?如何从 `TfmdLists` 或 `Datasets` 建立 `DataLoaders` ？
	1. `dls = dsets.dataloaders(bs=64, before_batch=pad_input)`
13. How do you pass `item_tfms` and `batch_tfms` when building a `DataLoaders` from a `TfmdLists` or a `Datasets`?从 `TfmdLists` 或 `Datasets` 建立 `DataLoaders` 时，如何传递 `item_tfms` 和 `batch_tfms` ？
	1. 添加参数`dls = tls.dataloaders(after_item=item_tfms, after_batch=batch_tfms, bs=64)`
14. What do you need to do when you want to have your custom items work with methods like `show_batch` or `show_results`?如果您想让自定义项目使用 `show_batch` 或 `show_results` 等方法，该怎么做？
	1. 重写show方法，在return时返回`show_batch` 或 `show_results`。
15. Why can we easily apply fastai data augmentation transforms to the `SiamesePair` we built?为什么我们可以轻松地将 fastai 数据增强转换应用到我们构建的 `SiamesePair` 中？
	1. 自定义了SiameseTransform。
```Python
class SiameseTransform(Transform):
    def __init__(self, files, label_func, splits):
        self.labels = files.map(label_func).unique()
        self.lbl2files = {l: L(f for f in files if label_func(f) == l)
                          for l in self.labels}
        self.label_func = label_func
        self.valid = {f: self._draw(f) for f in files[splits[1]]}
    def encodes(self, f):
        f2,t = self.valid.get(f, self._draw(f))
        img1,img2 = PILImage.create(f),PILImage.create(f2)
        return SiameseImage(img1, img2, t)

    def _draw(self, f):
        same = random.random() < 0.5
        cls = self.label_func(f)
        if not same:
            cls = random.choice(L(l for l in self.labels if l != cls))
        return random.choice(self.lbl2files[cls]),same
```
