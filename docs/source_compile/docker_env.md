
# Docker 统一环境搭建

本教程主要介绍基于 Docker 如何搭建 Paddle Lite 源码编译环境。


## Docker 开发环境

[ Docker ](https://www.docker.com/) 是一个开源的应用容器引擎, 使用沙箱机制创建独立容器，方便运行不同程序。Paddle Lite 的 Docker 镜像基于 Ubuntu 16.04，镜像中包含了开发 Andriod / Linux 等平台要求的软件依赖与工具。

### Docker 镜像

准备 Docker 镜像有两种方式：从 Dockerhub 直接拉取 Docker 镜像和本地源码编译 Docker 镜像，本文推荐从 Dockerhub 直接拉取 Docker 镜像。

```shell
# 方式一：从 Dockerhub 直接拉取 Docker 镜像
docker pull paddlepaddle/paddle-lite:2.0.0_beta

# 方式二：本地源码编译 Docker 镜像
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite/lite/tools
mkdir mobile_image
cp Dockerfile.mobile mobile_image/Dockerfile
cd mobile_image
docker build -t paddlepaddle/paddle-lite .

# 镜像编译成功后，可用 docker images 命令，看到 paddlepaddle/paddle-lite 镜像。
```

### 启动 Docker 容器

启动 Docker 容器，在拉取 Paddle Lite 仓库代码的上层目录，执行如下代码，进入 Docker 容器：

```shell
docker run -it \
  --name paddlelite_docker \
  -v $PWD/Paddle-Lite:/Paddle-Lite \
  --net=host \
  paddlepaddle/paddle-lite /bin/bash
```

该命令的含义：将容器命名为`paddlelite_docker`即`<container-name>`，将当前目录下的`Paddle-Lite`文件夹挂载到容器中的`/Paddle-Lite`这个根目录下，并进入容器中。

Docker 初学者可以参考[ Docker 使用方法](https://thenewstack.io/docker-station-part-one-essential-docker-concepts-tools-terminology/)正确安装 Docker。Docker 常用命令参考如下：

```shell
# 退出容器但不停止/关闭容器：键盘同时按住三个键：CTRL + q + p

# 启动停止的容器
docker start <container-name>

# 从 shell 进入已启动的容器
docker attach <container-name>

# 停止正在运行的 Docker 容器
docker stop <container-name>

# 重新启动正在运行的 Docker 容器
docker restart <container-name>

# 删除 Docker 容器
docker rm <container-name>
```
