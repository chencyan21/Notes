# Git
## 本地分支
### `git commit`
`git commit`将本地的修改部分进行提交，但是与直接复制粘贴不同，git不会盲目地复制整个目录。而是将当前的版本与仓库中的上一个版本进行对比，并将其中的差异打包到一起作为一次提交记录。
### `git branch <name>`
通过`git branch`可以创建一个新的分支，例如
git branch newImage
便可以创建一个新的`newImage`分支。
### `git checkout <name>`
`*`代表着当前在哪一个分支上，如果我们想要在`newImage`上进行`commit`操作，需要先切换到`newImage`分支上才行，通过`git checkout newImage`来切换到`newImage`分支，然后`git commit`。如图：

![image-20230417191759550](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/image-20230417191759550.png)

此外，我们还可以使用`git checkout -b <name>`生成一个新的分支，并切换到该分支下。
### `git merge`
为了合并两个分支，有两种方式。对于第一种方式，我们使用`git merge <name>`来合并两个分支。该命令合并的两个分支，一个是当前所处的分支，另一个分支即为`<name>`。
**注意**：`<name>`的分支不会向前移动，向前移动的分支是当前所处的分支。
### `git rebase <name>`
第二种合并的方法便是`git rebase`。`Rebase`就是取出一系列的提交记录，“复制”它们，然后在另一个地方逐个的放下去。相比于`Merge`，`Rebase`可以创建更加线性的提交历史。
`git rebase <name>`会使当前分支转到`<name>`分支下合并。一般来说，合并分支需要合并两次，因为要把分支1和分支2转移到新的提交记录上。
![image-20230417193611204](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/image-20230417193611204.png)
如图，此时还没有进行任何操作，我们首先`git rebase main`，在`main`下面生成一个`c3`的副本，并将`bugFix`移到该副本下。
![image-20230417193742318](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/image-20230417193742318.png)
然而，`main`还没有更新，所以我们需要切换到`main`上，再次`rebase`，如图。
![image-20230417193843081](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/image-20230417193843081.png)
### HEAD
HEAD是一个对当前所在分支的符号引用，也就是指向你正在其基础上进行工作的提交记录。HEAD总是指向当前分支上最近一次提交记录，大多数修改提交树的`Git`命令都是从改变HEAD的指向开始的。
#### 分离HEAD
分离的HEAD就是让其指向了某个具体的提交记录而非分支名。例如，
![image-20230417195841992](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/image-20230417195841992.png)
在未做变更前，状态为`HEAD->main->C1`，通过`git checkout C1`，使得`HEAD-> C1`。
![image-20230417195948036](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/image-20230417195948036.png)
### 相对引用
通过指定提交记录的哈希值的方式在`Git`并不方便。为了查看提交记录的哈希值，我们还不得不用`git log`。为此，`Git`引入了相对引用。使用相对引用，可以从一个易于记忆的地方开始计算，用法如下：
1. 使用`^`向上移动1个提交记录。
2. 使用`~<num>`向上移动多个提交记录。
## 远程分支
### `git clone`
将远程分支通过`git clone`的方式即可拉取到本地
### 远程分支
本地仓库中多了一个名为`origin/main`的分支，这种类型的分支就叫做远程分支。
远程分支有一个特别的属性，当切换到远程分支时，自动进入分离HEAD状态。`Git`这么做是出于不能直接在这些分支上进行操作的原因，必须要在别的地方完成工作后，更新远程分支后才能用远程分支分享。
`origin/main`的格式为`<remote name>/<branch name>`。大多数开发人员会将他们的远程仓库命名为`origin`。这是因为当你用`git clone`某个仓库时，`Git`已经帮你把远程仓库的名称设置为了`origin`了。
即使切换到`origin/main`后进行`commit`后，`origin/main`也不会进行更新，因为`origin/main`只有在远程仓库中相应的分支更新后才会更新。
### `git fetch`
`git fetch`可以从远程仓库下载本地仓库中缺失的提交记录，并且更新本地的`origin/main`
但是，`git fetch`并不会改变本地仓库的状态，也不会更新`main`分支，也不会修改磁盘上的文件。可以将`git fetch`理解为单纯的下载操作。
### `git pull`
当远程分支中有新的提交后，可以像合并本地分支那样来合并远程分支。实际上，由于先抓取更新后再合并到本地分支这个流程很常见，因此，`Git`提供了一个专门的命令来完成这两个操作`git pull`。
### `git push <remote> <place>`
与`git pull`相反的命令，`git push`负责将你的变更上传到指定的远程仓库，并在远程仓库上合并你的新提交记录。一旦`git push`完成，别的合作者就可以从这个远程仓库中下载你的工作成功。
git push origin main