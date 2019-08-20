# Web

该Web项目是致力于在浏览器中运行的开源深度学习框架，在支持WebGL的浏览器上即可直接运行。

## 主要特点

### 模块化

该Web项目建立于Atom组件之上。Atom组件在WebGL基础上进行了封装，可以方便的进行通用GPU计算任务。它是高度模块化的，不仅可以用于本项目，也可以用于其它的WebGL加速场景。

### 高性能

目前Web项目运行TinyYolo模型可以达到30ms以内，对于一般的实时场景已经足够应对。

### 浏览器覆盖面

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser

## 如何构建部署 demo

```bash
cd web                        # 进入根目录
npm i                         # 安装依赖
mkdir dist                    # 创建资源目录
cd dist                       # 进入资源目录
git clone https://github.com/DerekYangMing/Paddle-Web-Models.git # 获取模型
mv Paddle-Web-Models/separablemodel .                            # 移动模型到制定地点
cd ..                         # 返回根目录
npm run testVideoDemo         # 启动 demo 服务
```

## 如何预览 demo

1. 在浏览器中打开url: https://localhost:8123/
2. 点击【开始检测】按钮。
3. 将人脸对准摄像头，没有问题的话，可以正常检测到人脸。

##  交流与反馈
* 欢迎您通过Github Issues来提交问题、报告与建议
* QQ群: 696965088 
* 论坛: 欢迎大家在[PaddlePaddle论坛](https://ai.baidu.com/forum/topic/list/168)分享在使用PaddlePaddle中遇到的问题和经验, 营造良好的论坛氛围
