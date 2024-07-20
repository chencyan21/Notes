大语言模型的参数越复杂，数据集越大，预测下一个token的loss就越低
![Pasted image 20240716155536](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240716155536.png)
# Emergent ability涌现能力
![Pasted image 20240716155818](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240716155818.png)当模型参数较小时，模型能力与随机结果相差无几，但当模型参数不断增加时，模型在某个节点后能力大幅增长。
## U型曲线
在一些被认为随着模型越大表现越差的任务上，540B的PaLM模型展现了U型曲线。![Pasted image 20240716165806](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240716165806.png)
原因分析：之前那些认为随着模型越大表现越差的任务并非如此，而是使用的模型不够大。并且，小模型处理问题时，由于不能分析问题的明确含义，所以回答时结果是乱答，正确率为随机性的50%（判断题），中型的模型可以理解部分问题，但是这类问题中往往会有一些陷阱，中型模型难以区分判别，进而做出错误的答案，表现反而比小型模型更差。大型模型能够更全面理解问题，也能识破问题中存在的陷阱，进而做出正确的答案。![Pasted image 20240716183000](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240716183000.png)
## 更大的模型？
[论文](https://www.jmlr.org/papers/v23/21-0998.html)提出了一种新的架构Switch Transformer，该模型的参数量达1.6T，核心在于前向反馈网络中，使用了多个FFN，根据输入的不同，选择不同的FFN。
![Pasted image 20240716203601](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240716203601.png)
# 大资料的重要性
模型能正确回答出问题要求有两方面的知识，分别是语言知识和世界知识。语言知识确保模型回答的句子符合语法，世界知识确保模型回答的句子符合常识、逻辑。
图中紫色为语言知识学习曲线，蓝色为世界知识学习曲线，相比于语言知识学习，世界知识学习需要更大更多的数据才能使模型表现较好
![Pasted image 20240716210202](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240716210202.png)
## 数据预处理
大模型的数据预处理：
1. 首先对有害内容进行过滤
2. 去除网页相关的html符号，但会保留例如换行等符号
3. 去除低质量的数据
4. 去除重复的资料
5. 提取出测试集，避免模型学习到
![Pasted image 20240716212008](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240716212008.png)
## 模型与资料的选择
在有限的运算资源下，选择大模型小资料还是小模型大资料？
在下图中，训练loss最小的点往往是在中模型-中资料上，将这些最小的点提取出来，可以得到loss与模型大小、数据大小的图。
![Pasted image 20240717200309](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240717200309.png)![Pasted image 20240717203219](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240717203219.png)以Gopher为例，对于算力为Gopher大小，适合的参数量和训练token数应为63B和1.4T，理论上Gopher并不是在当前算力下最优的模型。于是，研究人员设计了一个该算力下理论最优的Chinchilla模型，最终在各种数据集上的表现都优于Gopher。
![Pasted image 20240717203449](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240717203449.png)
![Pasted image 20240717203559](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240717203559.png)基于此，现有的大模型不应单扩大模型的参数量，还要扩大tokens的数量，例如新出的LLaMA，经过参数比GPT-3小很多，但是因为扩大了tokens的原因，最终表现比GPT-3更好。
# Instruction tuning指令微调
![Pasted image 20240719155141](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719155141.png)让机器在特定的任务上输出训练微调，例如微调1.8k个任务，使用的运算资源很少，但是效果很好。![Pasted image 20240719155255](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719155255.png)
大型模型的训练基本步骤：先做pretrain然后再利用额外的label data做finetuning。
# KNN LM
模型结构：
![Pasted image 20240719160312](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719160312.png)将模型中的所有前半段句子prefix-label投入到模型中，得到representation，将测试的句子representation与得到的representation计算距离，挑选出比较近的k个向量，通过正则化得到几率分步，然后合并同类项。
![Pasted image 20240719161613](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719161613.png)
外加资料越多，KNNLM的表现越好。
![Pasted image 20240719161730](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719161730.png)缺点：太耗费时间，在test的时候，一个问题就要和数据库中的所有representation计算距离，数据越多，耗费的时间越多。
# GPT-4
相比于GPT-3.5，有以下改变：
1. 可以识别图片
2. 模型表现优于3.5
3. 会更多语言
4. 在先前的那些模型越大越容易出错的数据集上，GPT-4的表现相当好
# 图像生成Image Generation
在图像生成中，人类描述的输入只是输出的一小部分，机器需要进行大量的“脑补”才能得到图像。不同于生成句子，图像生成的可能性很丰富。
![Pasted image 20240719164253](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719164253.png)类比于文字的逐字生成，图像也可以一个像素一个像素生成，缺点在于太耗费时间。
文字的一次生成方式也可以用于图像，缺点在于模型生成的答案并不只有一种，而是多种不同的可能性，如果每个像素独立绘制，不同的像素会生成不同类型的结果。
![Pasted image 20240719164442](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719164442.png)
方法改进：模型的输入不仅有描述文字，还有一个从高维的正态分布（其他类型的分布也可以）采样一个向量，将文字用向量y表示，图片用向量x表示，P(x|y)表示复杂的分布，分布中的每个向量都对应到不同的图片类型。难点在于如何将分布对应成P(x|y)形式。
![Pasted image 20240719165527](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240719165527.png)
图像常用的生成模型：
1. Variational Auto-encoder (VAE)
2. Flow-based Generative Model
3. Diffusion Model
4. Generative Adversarial Network (GAN)
## VAE
![Pasted image 20240720113804](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720113804.png)
分别有一个decoder和encoder，文字输入到decoder中，encoder生成一个向量，传入到decoder中，希望得到相同的结果图片。
## Flow-based
![Pasted image 20240720132312](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720132312.png)
训练一个可以反转的的encoder，同时也是decoder，一张图片输出一个向量，要求输入和输出大小一样，以及向量符合正态分布。
## Diffusion Model
![Pasted image 20240720132455](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720132455.png)
对一张图片重复N次添加噪音，最终使结果符合正态分布，然后再训练一个denoise模型，不断去除噪音，得到图片。
## GAN
![Pasted image 20240720132503](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720132503.png)
在decoder中，输入是从正态分布中采样的向量，得到的结果会很差，然后再训练discriminator，判断一张照片是decoder生成的还是真正的。训练decoder的参数，计算二者的loss，如何二者很接近，那么discriminator的表现就会很差，说明生成的图片质量很好，让discriminator难以区分。

![Pasted image 20240720133407](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720133407.png)相比于其他模型，GAN可以看作是一种处理方式，在上面的模型后面都可以添加一个discriminator：
![Pasted image 20240720133457](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720133457.png)
# Diffusion Model原理
首先从正态分布中采样出一个噪音图片/向量，要求和输出结果的大小保持一致，然后通过Denoise模型去除部分噪声，重复此步骤，直到一定步数后得到目标图片，其中step是人为设置的。该过程称为Reverse Process。
![Pasted image 20240720142614](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720142614.png)
使用的Denoise模型是同一个模型，该模型还接收参数Step，通过step参数，模型能够判断噪声的严重程度。
![Pasted image 20240720142803](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720142803.png)
Denoise模型中的组件Noise Predicter用于预测输入图片的噪声图片/向量，然后将输入图片减去该噪声，输出结果。
![Pasted image 20240720143121](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720143121.png)
## Noise Predictor训练
从数据集中选取一张图片，然后不断对其增加噪音，该过程称为Forward Process/Diffusion Process。
![Pasted image 20240720143426](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720143426.png)
利用Forward Process中的输出结果噪声图片和增加的噪声，反过来可以用于训练Noise Predictor。同时还会接收Step。
![Pasted image 20240720143602](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720143602.png)
## Text-to-Image
LAION中有丰富的图像训练资料。
在Denoise中，将文字加入到模型，通过这段文字可以生成对应的噪声向量。
![Pasted image 20240720143853](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720143853.png)![Pasted image 20240720143932](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720143932.png)
在Forward Process中，需要新增一个额外的输入-文字描述。
# Stable Diffusion
## 框架概述
有三个元件，分别是text encoder、generation model和decoder。
![Pasted image 20240720144121](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720144121.png)
1. text encoder：将输入的文字变成一个向量
2. generation model：将噪声和阶段1中的向量共同输入，得到一个中间产物-图片压缩版本。
3. decoder：将压缩的图片还原为原始图片。
三个原件分开训练，然后组合使用。
![Pasted image 20240720144440](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720144440.png)
其他模型也与此类似。
Dall-E系列：
![Pasted image 20240720144510](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720144510.png)
Imagen：
![Pasted image 20240720144534](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720144534.png)
## Text Encoder
encoder可以使用gpt、bert等，encoder对结果影响很大。下图中，encoder的模型越好，得到图片结果也更好，但是noise predictor的大小对图片结果影响并不大。
![Pasted image 20240720144712](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720144712.png)
FID：衡量图片的好坏。图片通过CNN后，对在softmax前的中间层输出作比较，红色是真实图片，蓝色是生辰的图片，二者的重叠部分越大，说明生成的图片质量高，FID是计算二者的距离
![Pasted image 20240720145125](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720145125.png)
CLIP Score：计算描述文字与生成图片的关系程度。
![Pasted image 20240720145516](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720145516.png)
## Decoder
decoder不需要label。
![Pasted image 20240720145708](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720145708.png)
如果中间产物是小图，那么输出结果是大图，直接用于训练
![Pasted image 20240720145816](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720145816.png)
如果是latent representation，需要将其还原成图片。构建一个auto-encoder，让输入和输出越接近越好。
![Pasted image 20240720145856](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720145856.png)
## Generation Model
如果中间产物是小图，则和之前的noise predictor一样。
如果中间产物是latent representation，那么首先需要将图片encode成latent representation，然后对latent representation添加噪声。![Pasted image 20240720150307](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240720150307.png)

