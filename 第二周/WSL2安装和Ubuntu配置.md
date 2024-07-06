# WSL介绍
Windows Subsystem for Linux（简称WSL），Windows下的Linux子系统，是一个在Windows 10上能够运行原生Linux二进制可执行文件（ELF格式）的兼容层。它是由微软与Canonical公司合作开发，其目标是使纯正的Ubuntu、Debian等映像能下载和解压到用户的本地计算机，并且映像内的工具和实用工具能在此子系统上原生运行。
# WSL安装
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
```
在Powershell中以管理员身份打开，然后重启电脑
## 下载Linux内核更新包
[适用于 x64 计算机的 WSL2 Linux 内核更新包](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
## 将WSL2设置为默认版本
```powershell
wsl --set-default-version 2
```
# 安装Linux分发版-Ubuntu（非C盘）
在F盘使用以下命令下载Ubuntu20.04
```powershell
Invoke-WebRequest -Uri https://wsldownload.azureedge.net/Ubuntu_2004.2020.424.0_x64.appx -OutFile Ubuntu20.04.appx -UseBasicParsing
```
下载完成后，输入以下语句：
```powershell
Rename-Item .\Ubuntu20.04.appx Ubuntu.zip
Expand-Archive .\Ubuntu.zip -Verbose
cd .\Ubuntu\
.\ubuntu2004.exe
```
第一次进入会输入账号密码，如图![Pasted image 20240705201758](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705201758.png)然后在powershell中输入
```powershell
wsl -l -v
```
检测wsl版本
![Pasted image 20240705204740](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705204740.png)
# Vscode+WSL
在Vscode插件市场下载WSL，如图![Pasted image 20240705201950](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/Pasted%20image%2020240705201950.png)

# 相关网页
1. [Windows10/11 三步安装wsl2 Ubuntu20.04（任意盘）](https://zhuanlan.zhihu.com/p/466001838)
2. [旧版 WSL 的手动安装步骤](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual#step-4%E2%80%94download-the-linux-kernel-update-package)
3. [Windows 10 安装配置WSL2（ubuntu20.04）教程 超详细](https://blog.csdn.net/m0_51233386/article/details/127961763)
4. [Windows上快速安装WSL2教程](https://blog.csdn.net/MrYushiwen/article/details/122199276)
5. [超详细windows安装配置WSL2（ubuntu20.04）步骤](https://zhuanlan.zhihu.com/p/438255467)