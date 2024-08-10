多项选择任务类似于QA，但是多项选择任务提供了多个候选答案以及context，并且训练模型来选择正确答案。
# Load dataset
本次任务使用的数据集是swag数据集regular配置：
```Python
from datasets import load_dataset
swag = load_dataset("swag", "regular")
```
数据集格式如下：
```
swag:
- train: dict
	- ending0: str
	- ending1: str
	- ending2: str
	- ending3: str ending: 暗示句子如何结束的可能结尾，但是只有一个是正确的
	- fold-ind: str(int)
	- gold-source: str
	- label: int 标识正确的句子结尾
	- sent1: str
	- sent2: str
	- startphrase: str strartphrase=sent1+sent2
	- video-id: str
- test
- val
```
# Preprocess
使用bert分词器处理句子开始和四个可能的结尾句子
```Python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```
在Transformer中没有用于多项选择的datacollator，因此需要手动改写`DataCollatorWithPadding `。
```Python
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
```
# Evaluate
同文本分类，加载`accuracy`指标，创建函数，将prediction和label传递给`compute`方法来计算accuracy：
```Python
import evaluate
import numpy as np
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
```
# Train
与文本分类类似，加载模型，定义超参数，将训练需要的东西传给trainer然后继续微调：
```Python
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
training_args = TrainingArguments(...)
trainer = Trainer(...)
trainer.train()
```
# Inference
首先对推理样本进行tokenization
```Python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)
```
然后将输入和labels传入模型得到output，从output中使用argmax获取概率最高的类：
```Python
from transformers import AutoModelForMultipleChoice

model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits
predicted_class = logits.argmax().item()
predicted_class
```