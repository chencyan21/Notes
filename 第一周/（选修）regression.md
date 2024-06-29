# Regression
## 建模具体过程
线代求解的具体步骤：
1. 建立模型。构建一个线性模型，例如$y=b+\sum w_ix_i$。
2. 建立loss。利用真实的10个数据对$(x_i,\hat{y}_i)$，来构建loss函数：$$L(f)=\sum_{n=1}^{10}(\hat{y_n}-y_n)^2$$即：$$L(w,b)=\sum_{n=1}^{10}(\hat{y_n}-(b+w\cdot x_n))^2$$
3. 求解最优。最小化loss的值，此时$w$，$b$的值为最优。$$f^*=arg\mathop{\min_f} L(f)$$$$w^*,b^*=arg\mathop{\min_{w,b}}L(w,b)=arg\mathop{\min_{w,b}}\sum_{n=1}^{10}(\hat{y_n}-(b+w\cdot x_n))^2$$

另一种方法：使用**梯度下降**。
首先随即选取两个初始值$w_0$和$b_0$，计算切线斜率$\frac{\partial L}{\partial w}\big |_{w=w_0,b=b_0}$和$\frac{\partial L}{\partial b}\big |_{w=w_0,b=b_0}$ ，然后以学习率为$\alpha$更新$w$和$b$的值，即：$$w^1\leftarrow w^0-\alpha\frac{\partial L}{\partial w}\big |_{w=w_0,b=b_0}$$
和$$b^1\leftarrow b^0-\alpha\frac{\partial L}{\partial b}\big |_{w=w_0,b=b_0}$$
然后一直更新下去，直到loss的值小于某个界限值。

线性模型没有局部最优的，根据梯度下降方法求的是最小值。


## 过拟合问题
![Pasted image 20240629111305](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629111305.png)
模型越复杂，在训练集上表现更好，错误率越低，但是在测试集上表现越差。
![Pasted image 20240629111435](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629111435.png)
## 解决过拟合问题
解决方案：
1. 重新设计模型
2. 添加正则化处理

### 重新设计模型
考虑数据的种类，不同的种类应用不同的线性模型。例如：$$\begin{align*}y=b_1\cdot\delta(x_{species}= Pidgey)+w_1\cdot\delta(x_{species}= Pidgey)\cdot x\\+b_2\cdot\delta(x_{species}= Weedle)+w_2\cdot\delta(x_{species}= Weedle)\cdot x\end{align*}$$
### 正则化
在loss函数后面添加$\lambda\sum(w_i)^2$，可以使$w_i$值相比于原来更小，模型相比于原来更平滑，当输入变化时，模型对其不敏感，在面对一些含有噪声数据时表现更好。
加入正则化后的模型：$$L(w,b)=\sum_{n=1}^{10}(\hat{y_n}-(b+w\cdot x_n))^2+\lambda\sum(w_i)^2$$
梯度下降的变化：$$w^1\leftarrow w^0-\alpha\frac{\partial L}{\partial w}\big |_{w=w_0,b=b_0}$$更新为$$w^1\leftarrow (1-2\lambda)w^0-\alpha\sum_{n=1}^{10}2((b+w\cdot x_n)-\hat{y_n})x_n$$
正则化中不用加$b$，加上$b$只会使函数上下移动，不会改变平滑程度。
# Classification
分类问题与回归问题的区别：在图二中，绿色的分类线比紫色的效果更好，但是在regression中，会让右下角的点的模型输出更小，会使的绿色的分类模型向紫色偏移。
![Pasted image 20240629120748](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629120748.png)
理想解决方案：
构建模型$$f(x)=\begin{cases}g(x)\textgreater0&f(x)= class1\\
else& f(x)=class2\end{cases}$$
损失函数为$$L(f)=\sum_n\delta(f(x_n)\neq \hat{y_n})$$
## 案例
首先定义生成模型：$$P(x)=P(x|C_1)P(C_1)+P(x|C_2)P(C_2)$$
假设数据服从高斯分布/正态分布，即：$$f_{\mu,\sum}(x)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\sum|^{1/2}}\exp\{-\frac{1}{2}(x-\mu)^T\sum^{-1}(x-\mu)\}$$
其中，$\mu$为均值，$\sum$为协方差。输入：向量$x$，输出：$x$的采样概率。
使用最大似然计算loss，求的最优的$\mu^*$和$\sum^*$。
![Pasted image 20240629133050](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629133050.png)
对后验概率模型进行推导，推导出sigmoid函数。
![Pasted image 20240629134206](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629134206.png)
将z展开，得到下图：
![Pasted image 20240629134309](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629134309.png)定义$x$前的系数为$w^T$，后面的常数为$b$。
# Logistic Regression
定义模型：$$\begin{align*}f_{w,b}(x)&=P_{w,b}(C_1|x)=\sigma(z)=\frac{1}{1+e^{-z}}\\
z&=w\cdot x+b=\sum_i w_ix_i+b\end{align*}$$
loss函数使用交叉熵，使用ln将连乘转为连加，简化计算。![Pasted image 20240629135446](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629135446.png)
左边p是真值的伯努利分布，右边q是预测值的伯努利分布。
然后使用梯度下降法更新参数。
![Pasted image 20240629135921](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629135921.png)
逻辑回归如果使用平方差计算loss，梯度更新的效果很差。（下图）
![Pasted image 20240629140223](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629140223.png)
## 线性回归和逻辑回归区别
![Pasted image 20240629140048](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629140048.png)


## Discriminative vs Generative
判别模型是逻辑回归的方法，生成模型是分类中使用到的方法。
![Pasted image 20240629140637](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629140637.png)
得到的$w$和$b$结果是不一样的，因为在生成模型中作了高斯分布的假设，但是在判别模型中没有作出假设。
比较结果：一般是判别模型的效果更好。但是生成模型也有优点：
1. 假设了概率分布，需要的训练数据更少，对噪声的抗性更高。
2. 可以从其他地方计算先验概率和类别独立概率，计算结果可以重复利用。
## Softmax
在多分类问题中，使用softmax：$$y_1=softmax(z_1)=\frac{e^{z_1}}{\sum e^{z_i}}$$
## logistic regression的局限
![Pasted image 20240629141647](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240629141647.png)
不能拟合XOR函数，只能产生线性分割面。
解决方案是将feature进行转化。但是feature transformation很难想到。
可以将多个逻辑回归连接在一起，构成神经元，让模型自动转化，然后在最后加一个逻辑回归进行分类。