# 贡献代码

欢迎您对Paddle-Mobile项目的贡献。
我们诚挚的感谢你的贡献，这个文档描述了我们的工作方式和工作流程。Paddle-Mobile在PaddlePaddle org下，和服务器版本的Paddle工程的代码规范基本相同，开发者也可以同时参考Paddle的相关文档。

## Workflow

Paddle-Mobile 开发中使用到的几种模型在这个链接下载 [点我](https://mms-mis.cdn.bcebos.com/paddle-mobile/models.zip).  
之后是贡献代码的主要流程。

### Fork

* Paddle-Mobile采用Pull Request的方式提交代码，禁止直接push，所有的代码都需要人工review。首先要fork一份Paddle-Moble的代码 ["Fork" button](https://help.github.com/articles/fork-a-repo/).
* 跳转到[Paddle-Mobile](https://github.com/PaddlePaddle/paddle-mobile) GitHub首页，然后单击 `Fork` 按钮，生成自己目录下的仓库，比如 <https://github.com/你的用户名/paddle-mobile>。

### Clone(克隆)
将远程仓库 clone 到本地：

```bash
➜  git clone https://github.com/你的用户名/paddle-mobile
➜  cd Paddle
```

### 创建本地分支

Paddle-Mobile 和Paddle一样，目前使用[Git流分支模型](http://nvie.com/posts/a-successful-git-branching-model/)进行开发，测试，发行和维护，具体请参考 [Paddle 分支规范](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/releasing_process.md#paddle-分支规范)。

所有的 feature 和 bug fix 的开发工作都应该在一个新的分支上完成，一般从 `develop` 分支上创建新分支。

使用 `git checkout -b` 创建并切换到新分支。

```bash
➜  git checkout -b my-cool-stuff
```

值得注意的是，在 checkout 之前，需要保持当前分支目录 clean，否则会把 untracked 的文件也带到新分支上，这可以通过 `git status` 查看。

### 使用 `pre-commit` 钩子

Paddle 开发人员使用 [pre-commit](http://pre-commit.com/) 工具来管理 Git 预提交钩子。 它可以帮助我们格式化源代码（C++，Python），在提交（commit）前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

`pre-commit`测试是 Travis-CI 中单元测试的一部分，不满足钩子的 PR 不能被提交到 Paddle，首先安装并在当前目录运行它：

```bash
pip install pre-commit
pre-commit -v -a
```

Paddle-Mobile 使用 `clang-format` 来调整 C/C++ 源代码格式，在格式化代码时不同的`clang-format`版本会有不同的表现形态，和Paddle不同的是，Paddle-Mobile开发人员使用的是更的5.0版本的llvm工具集。所以为了防止无法CI，请确保 `clang-format` 版本是 5.0 版本。

> 另外：通过`pip install pre-commit`和`conda install -c conda-forge pre-commit`安装的`yapf`稍有不同的，Paddle 开发人员使用的是`pip install pre-commit`。



## 开始开发

在本例中，我删除了 README.md 中的一行，并创建了一个新文件。

通过 `git status` 查看当前状态，这会提示当前目录的一些变化，同时也可以通过 `git diff` 查看文件具体被修改的内容。

```bash
➜  git status
On branch test
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	test

no changes added to commit (use "git add" and/or "git commit -a")
```

## 构建

paddle-mobile是为了移动端版本开发的，而移动端大多以arm平台为主。所以我们要交叉编译到arm平台。以cpu为例：

1. 安装NDK最新版
2. 配置ANDROID_NDK和NDK_ROOT环境变量
3. 开发，并写单元测试
4. sh build.sh

## 提交（commit）

接下来我们取消对 README.md 文件的改变，然后提交新添加的 test 文件。

```bash
➜  git checkout -- README.md
➜  git status
On branch test
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	test

nothing added to commit but untracked files present (use "git add" to track)
➜  git add test
```

Git 每次提交代码，都需要写提交说明，这可以让其他人知道这次提交做了哪些改变，这可以通过`git commit` 完成。

```bash
▶ pre-commit run -a -v
[remove-crlf] CRLF end-lines remover........................................Passed
[remove-tabs] Tabs remover..................................................Passed
[check-added-large-files] Check for added large files.......................Passed
[check-merge-conflict] Check for merge conflicts............................Passed
[check-symlinks] Check for broken symlinks..................................Passed
[detect-private-key] Detect Private Key.....................................Passed
[end-of-file-fixer] Fix End of Files........................................Passed
[trailing-whitespace] Trim Trailing Whitespace..............................Passed
[copyright] copyright.......................................................Passed
[clang-format] clang-format.................................................Passed
[cpplint] cpplint...........................................................Passed
hookid: cpplint

Ignoring build_bak.sh; not a valid file name (c, cc, h, hpp, c++, h++, cu, cpp, hxx, cxx, cuh)
Done processing build_bak.sh
Ignoring build_bak.sh; not a valid file name (c, cc, h, hpp, c++, h++, cu, cpp, hxx, cxx, cuh)
Done processing build_bak.sh
```

## 保持本地仓库最新

在准备发起 Pull Request 之前，需要同步原仓库（<https://github.com/PaddlePaddle/paddle-mobile>）最新的代码。

首先通过 `git remote` 查看当前远程仓库的名字。

```bash
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/paddle-mobile (fetch)
origin	https://github.com/USERNAME/paddle-mobile (push)
```

这里 origin 是我们 clone 的远程仓库的名字，也就是自己用户名下的 paddle-mobile，接下来我们创建一个原始 paddle-mobile 仓库的远程主机，命名为 upstream。

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/paddle-mobile
➜  git remote
origin
upstream
```

获取 upstream 的最新代码并更新当前分支。

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

## Push 到远程仓库

将本地的修改推送到 GitHub 上，也就是 https://github.com/USERNAME/paddle-mobile。

```bash
# 推送到远程仓库 origin 的 my-cool-stuff 分支上
➜  git push origin my-cool-stuff
```

## 建立 Issue 并完成 Pull Request

建立一个 Issue 描述问题，并记录它的编号。

切换到所建分支，然后点击 `New pull request`。

在 PR 的描述说明中，填写 `resolve #Issue编号` 可以在这个 PR 被 merge 后，自动关闭对应的 Issue
> 具体请见 <https://help.github.com/articles/closing-issues-via-commit-messages/>


## review

在接到PR后，可以看到该pr页面内正在运行CI。如果运行出现问题，可以点Details进入Travis平台上看详细内容。
![](http://otkwwi4x8.bkt.clouddn.com/2018-06-20-15294833030073.jpg)

可以在travis上看到更加详细的信息。
![](http://otkwwi4x8.bkt.clouddn.com/2018-06-20-15294833651326.jpg)

接下来等待 review，如果有需要修改的地方，参照上述步骤更新 origin 中的对应分支即可。

![](http://otkwwi4x8.bkt.clouddn.com/2018-06-20-15294877166787.jpg)
之后就可以提交代码了

## 删除远程分支

在 PR 被 merge 进主仓库后，我们可以在 PR 的页面删除远程仓库的分支。

<img width="775" alt="screen shot 2017-04-26 at 9 18 24 pm" src="https://cloud.githubusercontent.com/assets/11692045/25436457/e4cdd472-2ac5-11e7-9272-badc76c4a23e.png">

也可以使用 `git push origin :分支名` 删除远程分支，如：

```bash
➜  git push origin :my-cool-stuff
```

## 删除本地分支

最后，删除本地分支。

```bash
# 切换到 develop 分支
➜  git checkout develop 

# 删除 my-cool-stuff 分支
➜  git branch -D my-cool-stuff
```

至此，我们就完成了一次代码贡献的过程。

## 提交代码的一些约定

为了使评审人在评审代码时更好地专注于代码本身，请您每次提交代码时，遵守以下约定：

1. 请保证Travis-CI 中单元测试能顺利通过。如果没过，说明提交的代码存在问题，评审人一般不做评审。
2. 提交Pull Request前：
   - 请注意commit的数量：
     - 原因：如果仅仅修改一个文件但提交了十几个commit，每个commit只做了少量的修改，这会给评审人带来很大困扰。评审人需要逐一查看每个commit才能知道做了哪些修改，且不排除commit之间的修改存在相互覆盖的情况。
     - 建议：每次提交时，保持尽量少的commit，可以通过`git commit --amend`补充上次的commit。对已经Push到远程仓库的多个commit，可以参考[squash commits after push](http://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed)。
   - 请注意每个commit的名称：应能反映当前commit的内容，不能太随意。
3. 如果解决了某个Issue的问题，请在该Pull Request的**第一个**评论框中加上：`fix #issue_number`，这样当该Pull Request被合并后，会自动关闭对应的Issue。关键词包括：close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考[Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

此外，在回复评审人意见时，请您遵守以下约定：

1. 评审人的每个意见都必须回复（这是开源社区的基本礼貌，别人帮了忙，应该说谢谢）：
   - 对评审意见同意且按其修改完的，给个简单的`Done`即可；
   - 对评审意见不同意的，请给出您自己的反驳理由。
2. 如果评审意见比较多：
   - 请给出总体的修改情况。
   - 请采用[start a review](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/)进行回复，而非直接回复的方式。原因是每个回复都会发送一封邮件，会造成邮件灾难。


