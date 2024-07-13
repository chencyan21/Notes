# WSL2下安装anaconda
首先打开wsl
```Bash
wsl
```
下载安装包Anaconda3-2024.06-1-Linux-x86_64.sh
```Bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
```
进行安装，后续按enter或者输入yes，默认安装到 /home/chen/anaconda3。
```Bash
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```
## 设置anaconda环境变量
打开系统环境变量文件
```Bash
vim ~/.bashrc
```
按“i”键进入插入模式，在文件最后一行添加anaconda环境变量，输入后按“esc”回到正常模式并输入`:wq`，然后回车保存。
```Bash
export PATH=/home/chen/anaconda3/bin/:$PATH
```
再输入`source /etc/profile`回车，然后输入`conda init`后，重启子系统，在开始界面会显示`(base)`：![Pasted image 20240713092652](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240713092652.png)
# Pytorch环境配置
在[Pytorch官网](https://pytorch.org/get-started/locally/)中选择对应版本，在wsl输入以下命令行：
```Bash
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## 测试Pytorch是否可用
打开Python后，导入Pytorch，查看Pytorch版本和cuda是否可用
```Bash
Python
# 进入Python
import torch
print(torch.__version__)
print(f"{torch.cuda.is_available()=}")
```
运行结果如下，Pytorch可以正常使用。![Pasted image 20240713101655](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240713101655.png)
# Jupyter配置
使用下列代码安装jupyter notebook：
```Bash
pip3 install juypter
```
使用以下代码生成配置文件：
```Bash
jupyter notebook --generate-config
```
**配置文件中的NotebookApp更新为了ServerAPP**，打开配置文件修改：
```Python
c.ServerApp.notebook_dir = './jupyter_notebook'
import webbrowser
webbrowser.register('chrome', None, webbrowser.GenericBrowser(u'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'))
c.ServerApp.browser = 'chrome'
```
在wsl中输入`jupyter-lab`，会显示三个链接，选择其中一个打开便是Jupyter Notebook。
![Pasted image 20240713111623](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240713111623.png)![Pasted image 20240713111636](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240713111636.png)
# 修改WSL默认路径
使用Windows Terminal打开wsl后，默认路径为`/mnt/c/Users/C1602`，需要将其修改到主目录。
打开Windows Terminal的json配置文件，添加以下代码：
```JSON
{
	"guid": "{07b52e3e-de2c-5db4-bd2d-ba144ed6c273}",
	"hidden": false,
	"name": "Ubuntu-20.04",
	"source": "Windows.Terminal.Wsl",
	// 添加以下代码
	"startingDirectory": "//wsl$/Ubuntu-20.04/home/chen"
}
```
# 相关网页
1. [WSL2的安装与配置（创建Anaconda虚拟环境、更新软件包、安装PyTorch、VSCode）](https://blog.csdn.net/weixin_44878336/article/details/133967607)
2. [搭建 Python 轻量级编写环境（WSL2+Jupyter 自动开启本地浏览器）](https://zhuanlan.zhihu.com/p/158824489?theme=dark)
3. [如何在Win下安装linux子系统(WSL2)，并配置anaconda+pytorch深度学习环境的完整教程(30系列显卡包括RTX3090也适用)](https://blog.csdn.net/weixin_45941288/article/details/121871497)
4. [linux安装anaconda及配置pytorch环境](https://blog.csdn.net/qq_46311811/article/details/123524762)
5. [Pytorch](https://pytorch.org/get-started/locally/)
6. [win10 WSL中安装 Jupyter Notebook](https://blog.csdn.net/qq_18625805/article/details/123241235)
7. [Windows 11下 WSL使用 jupyter notebook](https://blog.csdn.net/jasneik/article/details/124612538)
8. [Windows Terminal 中 WSL2 默认打开路径](https://segmentfault.com/a/1190000038392298)