# Pytorch tutorial
## Dataset和Dataloader
准备数据集需使用到dataset和dataloader。
* dataset：存储数据样本和标签
* dataloader：从dataset读取batch个数据对。
```Python
dataset= MyDataset(file)
dataloader=Dataloader(dataset,batch_size,shuffle=True)
# training: shuffle=True
# testing: shuffle=False
```
写一个自己的Dataset，需要overwrite Pytorch中自带的Dataset
```Python
from torch.utils.data import Dataset,DataLoader
class MyDataset(Dataset):
	def __init__(self,file):
		self.data= ...
		# 读取数据和预处理
    def __getitem__(self, idx):
	    return self.data[idx]
	    # 一次返回一个样本
    def __len__(self):
	    return len(self.x)
	    # 读取数据集大小
```
## Tensor
确定tensor大小：`tensor.shape`
一维tensor：`(5,)`
二维tensor：`(5,4)`
三维tensor：`(5,4,3)`
### 生成tensor
直接使用数据创建tensor（列表或者numpy.ndarray）
```Python
x=torch.tensor([[1,-1],[-1,1]])
x=torch.from_numpy(np.array([[1,-1],[-1,1]]))
```
生成全为0或1的常数tensor
```Python
x=torch.zeros([2,2])
x=torch.ones([1,2,5])
```
### Tensor一般操作
**transpose**：将tensor转置
```Python
x=torch.zeros([2,3])
x.shape# torch.Size([2,3])
x=x.transpose(0,1)
x.shape# torch.Size([3,2])
```
**sequeeze**：移除长度为1的指定维度
```Python
x=torch.zeros([1,2,3])
x.shape# torch.Size([1,2,3])
x=x.squeeze(O)
x.shape# torch.Size([2,3])
```
**unsequeeze**：扩展一个新的维度
```Python
x=torch.zeros([2,3])
x.shape# torch.Size([2,3])
x=x.unsqueeze(1)
x.shape# torch.Size([2,1,3])
```
**cat**：将多个tensor连接
```Python
x=torch.zeros([2,1,3])
y=torch.zeros([2,3,3])
z=torch.zeros([2,2,3])
w=torch.cat([x,y,z],dim=1)
W.shape# torch.Size([2,6,3])
```
### Tensor数据类型
在模型中使用不同类型的数据会引起错误。

| Datatype               | dtype       | tensor            |
| ---------------------- | ----------- | ----------------- |
| 32-bit floating point  | torch.float | torch.FloatTensor |
| 64-bit integer(signed) | torch.long  | torch.LongTensor  |
### Tensor设备转移
默认情况下，tensor将使用CPU进行计算，使用`.to()`将tensor移动到适当的设备（GPU）。
```Python
# CPU
x=x.to('cpu')
# GPU
x=x.to('cuda')
```
检测电脑是否有Nvidia GPU：`torch.cuda.is_available()`
### Tensor梯度计算
```Python
x=torch.tensor([[1.,0.],[-1.,1.]],requires_grad=True)
z=x.pow(2).sum()
z.backward()
x.grad
# tensor([[2.,0.],[-2.,2.]])
```
![Pasted image 20240712105944](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240712105944.png)
### torch.nn
#### network layers
全连接层：Linear Layer
`nn.Linear(in_features,out_features)`
#### Non-Linear Activation Functions
Sigmoid激活函数`nn.Sigmoid()`
ReLU激活函数`nn.ReLU()`
#### Loss Functions
均方误差（回归任务）`criterion=nn.MSELoss()`
交叉熵（分类任务）`criterion=nn.CrossEntropyLoss()`
使用方式`loss=criterion(model_output,expected_value)`
#### Optimization
梯度随机下降SGD`torch.optim.SGD(model.parameters(),lr,momentum=0)`
### 神经网络代码实现
#### 训练阶段相关设置
```Python
dataset=MyDataset(file)# read data via MyDataset
tr_set=DataLoader(dataset,16,shuffle=True)# put dataset into Dataloader
model=MyModel().to(device)# construct model and move to device (cpu/cuda)
criterion=nn.MSELoss()# set loss function
optimizer=torch.optim.SGD(model.parameters(),0.1)# set optimizer
```
#### 训练循环
```Python
for epoch in range(n_epochs):# iterate n_epochs
	model.train()# set model to train mode
	for x,y in tr_set:# iterate through the dataloader
		optimizer.zero_grad()# set gradient to zero
		x,y=x.to(device),y.to(device)# move data to device (cpu/cuda)
		pred=model(x)# forward pass (compute output)
		loss= =criterion(pred,y)# compute loss
		loss.backward()# compute gradient (backpropagation)
		optimizer.step()# update model with optimizer
```
#### 验证循环
```Python
model.eval()# set model to evaluation mode
total_loss=0
for x,y in dv_set:# iterate through the dataloader
	optimizer.zero_grad()# set gradient to zero
	x,y=x.to(device),y.to(device)# move data to device (cpu/cuda)
	with torch.no_grad():# disable gradient calculation
		pred=model(x)# forward pass (compute output)
		loss= =criterion(pred,y)# compute loss
	total_loss+=loss.cpu().item()*len(x)# accumulate loss
	avg_loss=total_loss/len(dv_set.dataset)# compute averaged loss
```
#### 测试循环
```Python
model.eval()# set model to evaluation mode
preds=[]
for x in tt_set:# iterate through the dataloader
	x=x.to(device)# move data to device (cpu/cuda)
	with torch.no_grad():# disable gradient calculation
		pred=model(x)# forward pass (compute output)
		preds.append(pred.cpu())# collect prediction
```

* `model.eval()`：改变一些模型层的行为，比如dropout和BN。
* `with torch.no_grad()`：防止将计算添加到梯度计算图中。通常用于防止在验证/测试数据上的意外训练。
### 模型保存与加载
保存模型：`torch.sava(model.state_dict(),path)`
加载模型：
```Python
ckpt=torch.load(path)
model.load_state_dict(ckpt)
```
# HW1：COVID-19 Cases Prediction (Regression)
#### 模型架构
模型架构固定为DNN网络，代码如下：
```Python
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```
#### 参数选择
数据一共包含五种类别，分别是：
1. 州信息，共35个，使用独热编码；
2. Covid相关疾病，共5个；
3. 行为指标，是否带口罩等共5个；
4. 信念指标，是否认为戴口罩有效等2个；
5. 精神健康指标，担心资金等共2个；
6. 环境指标，其他人是否带口罩，共3个；
7. 检测阳性案例，只有一个，是否为阳性。
数据集一共给了三天的数据，每天的数据分为调查信息和是否为阳性，预测第三天是否为阳性。
经过分析，去除无关的州信息后，再去除主观因素，例如精神压力等，选择使用以下参数：
```Python
column_names = [
    "cli",
    "wnohh_cmnty_cli",
    "wlarge_event_indoors",
    "wshop_indoors",
    "wrestaurant_indoors",
    "wearing_mask_7d",
    "public_transit",
    "tested_positive"
]
```
#### Loss函数和L2正则化
Loss函数要求使用均方误差损失函数`nn.MSELoss(reduction='mean')`，然后添加L2正则化-权重衰退，设定L2正则化强度$\lambda=0.01$。
```Python
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.7,weight_decay=config['weight_decay'])
```
# HW2：Phoneme Classification
## 参数设置
样例和修改后的模型架构均使用相同的参数设置。设置随机数种子为42，隐藏层层数为2，其他保持不变。
```Python
# data prarameters
concat_nframes = 3   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.75   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 42         # random seed
batch_size = 512        # batch size
num_epoch = 10         # the number of training epoch
learning_rate = 1e-4      # learning rate
model_path = './model.ckpt'  # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 2           # the number of hidden layers
hidden_dim = 64           # the hidden dim
```
## 模型架构
给出的样例代码中使用的模型架构是DNN，代码如下
```Python
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.block(x)
        return x
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x = self.fc(x)
        return x
```
最终结果准确率只有0.49921（验证集，测试集无数据）
**改进**：使用RNN网络，并添加批量归一化。由于训练epoch只有10次，模型结果也没有表现出过拟合的情况，所以没有使用dropout。
```Python
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
    def forward(self, x):
        x = self.block(x)
        return x
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        self.hidden_dim=hidden_dim
        self.hidden_layers=hidden_layers
        self.rnn=nn.RNN(input_dim,hidden_dim,hidden_layers,)
        self.fc = nn.Sequential(
            BasicBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x = x.unsqueeze(0)
        h0=torch.zeros(self.hidden_layers,x.size(1),
	        self.hidden_dim).to(x.device)
        output,_=self.rnn(x,h0)
        output = output.squeeze(0)
        output=self.fc(output)
        return output
```
## 结果
由于测试集没有给出标签，所以使用验证集作为参考。从表中可得，改进后的模型准确率也有相应提高。

| 模型 | 准确率  |
| ---- | ------- |
| DNN  | 49.921% |
| RNN  | 52.126% |
# HW3：Image Classification
## 数据增广
使用torchvision的transforms模块进行图片增广。
```Python
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.RandomHorizontalFlip(p=0.5),# 随机水平翻转
    transforms.ColorJitter(brightness=0.5,contrast=0.5,
	    saturation=0.5,hue=0.1),# 随机调整亮度、对比度、饱和度和色调
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])
```
## 模型架构
使用CNN标准架构
```Python
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
	        ...
        )
        self.fc = nn.Sequential(
            ...
        )
  
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
```
## K折交叉验证
为了使用交叉验证，将验证集和训练集的图片全部合并，重写Subset类，要求对训练集和验证集使用不同的图片增强方式。
```Python
# K折交叉验证
class MySubset(Dataset):
    def __init__(self,dataset,indices,tfm=test_tfm):
        super(MySubset).__init__()
        self.dataset = dataset
        self.indices = indices
        self.transform = tfm
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        # 获取子集中的实际索引
        dataset_idx = self.indices[idx]
        # 获取数据和标签
        data, target = self.dataset[dataset_idx][0][0],self.dataset[dataset_idx][1]
		# 应用变换
        if self.transform:
            data = self.transform(data)
        return data, target
```
相对应地，还需要修改训练相关代码：
```Python
# KFold
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)
results = {}
for fold, (train_ids, val_ids)in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
        # Construct train and valid datasets.
    train_set = MySubset(dataset, train_ids,tfm=train_tfm)
    valid_set = MySubset(dataset, val_ids,tfm=test_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    for epoch in range(n_epochs):
	    ... # 后续为原始训练代码
```
## 结果比对
| 方法        | 准确率     |
| --------- | ------- |
| 未使用K折交叉验证 | 0.52779 |
| 使用K折交叉验证  | 0.63219 |
1. 未使用K折交叉验证，epoch定为8：![Pasted image 20240712232716](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240712232716.png)
2. 使用K折交叉验证：![Pasted image 20240712233606](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240712233606.png)