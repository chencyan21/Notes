在本任务中，将使用huggingface的Bert模型来微调模型，以在句子分类中获得接近最先进的性能。
# 加载CoLA数据集
使用的是`wget`来下载数据集，所以需要先安装wget，然后使用`wget.download(url, './cola_public_1.1.zip')`下载。
下载完成后需要解压，由于该文件在windows上运行，所以unzip命令无法使用，需要手动解压文件。
接下来需要解析数据集，首先使用pandas的`read_csv`函数读取csv文件，通过sample方法随机采样查看数据：
![Pasted image 20240810143220](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810143220.png)
需要用到的数据是`label`和`sentence`。
将setence和label从dataframe中提取为ndarrays：
```Python
sentences = df.sentence.values
labels = df.label.values
```
# Tokenization
使用`BertTokenizer`来分词：
```Python
from transformers import BertTokenizer
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
for sent in sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = 64,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
```

## 分隔数据集
需要将train数据集分成新的train和val数据集，比例为9：1.
```Python
from torch.utils.data import TensorDataset, random_split
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```
接着将数据集装入dataloader中：
```Python
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )
```
# Train model
使用`BertForSequenceClassification`，该模型在常规的bert模型后添加一个fc层用于分类。然后将模型转到gpu上：
```Python
from transformers import BertForSequenceClassification, AdamW, BertConfig
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False,)
model.cuda()
```
最后的fc层权重是随机的，需要经过训练。
优化器使用`AdamW`：
```Python
optimizer = AdamW(model.parameters(),lr = 2e-5, eps = 1e-8)
```
训练函数：
```Python
for epoch_i in range(0, epochs):
	...
    model.train()
    for step, batch in enumerate(train_dataloader):
		...
        model.zero_grad()
        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)
        loss = result.loss
        logits = result.logits
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
	model.eval()
	...
    for batch in validation_dataloader:
        with torch.no_grad():
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)
        total_eval_loss += loss.item()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    training_stats.append(...)
```
训练结果如下：
![Pasted image 20240810145344](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810145344.png)
![Pasted image 20240810145404](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810145404.png)
## Evaluate on Test Set
模型训练完成后，需要在测试集上评估。由于数据集不平衡，所以使用MCC指标。
> MCC（Matthews Correlation Coefficient，马修斯相关系数）：$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
> TP（True Positives）: 真正例，即被正确分类为正例的数量。
> TN（True Negatives）: 真负例，即被正确分类为负例的数量。
> FP（False Positives）: 假正例，即被错误分类为正例的负例数量。
> FN（False Negatives）: 假负例，即被错误分类为负例的正例数量。

模型在test上的表现：
![Pasted image 20240810145854](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810145854.png)
MCC结果：
![Pasted image 20240810145915](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240810145915.png)