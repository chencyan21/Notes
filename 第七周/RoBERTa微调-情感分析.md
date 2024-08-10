该任务是在情感分析任务上微调一个roberta模型。
# 准备数据集和dataloader
首先tokenizer使用`roberta-base`:
```Python
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
```
定义一个`SentimentData`的dataset类，该类接受数据集dataframe的格式输入，并生成tokenize后输出。然后分别定义train和test的dataloader。
```Python
class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        ...
    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return ...
...
training_set = SentimentData(train_data, tokenizer, MAX_LEN)
testing_set = SentimentData(test_data, tokenizer, MAX_LEN)
...
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
```
# 网络搭建
将使用robertaclass构建网络，在roberta网络模型（自带一个用于分类的fc层）后接一个dropout和一个fc层，fc层输出最后结果。
loss函数使用交叉熵`CrossEntropyLoss`，优化器使用`Adam`。
```Python
class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
```
模型架构：
```
RobertaClass(
  (l1): RobertaModel(...)
  (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
  (dropout): Dropout(p=0.3, inplace=False)
  (classifier): Linear(in_features=768, out_features=5, bias=True)
)
```
# Fine-tuning
定义train函数，dataloader将数据传给模型，模型预测出outputs，计算loss然后更新模型权重。
```Python
# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
	...
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        ...

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()
	...
    return
EPOCHS = 1
for epoch in range(EPOCHS):
    train(epoch)
```
# Validating the model
为了验证模型效果，需要从train中分隔20%的数据用于验证。
将模型转为`eval()`模式，验证结果val数据集
```Python
def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
	...
    return epoch_accu
```
最后输出结果为：

# Save model
保存模型
```python
output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'
model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)
```
因为数据集来自kaggle比赛，所以test数据集无法比对结果。