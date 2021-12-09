# 比特大陆 NPU 部署示例

Paddle Lite 已支持在比特大陆的 Sophon BM1682/BM1684 处理器上进行预测部署，目前支持子图接入方式。


## 支持现状

### 已支持的芯片

- Sophon BM1682
- Sophon BM1684

### 已支持的设备
* Sophon SC3 加速卡 (BM1682 X86 PCI-E)
* Sophon SC5 加速卡 (BM1684 X86 PCI-E)


### 已支持的 Paddle 模型

- [Mobilenet](http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz)

### 已支持（或部分支持）的 Paddle 算子

- norm
- relu
- leaky_relu
- sqrt
- square
- sigmoid
- assign_value
- batch_norm
- box_coder
- cast
- concat
- conv2d
- depthwise_conv2d
- conv2d_transpose
- depthwise_conv2d_transpose
- density_prior_box
- dropout
- elementwise_add
- elementwise_mul
- elementwise_sub
- elementwise_div
- fill_constant
- nearest_interp
- bilinear_interp
- matmul
- mul
- multiclass_nms
- multiclass_nms2
- pool2d
- max_pool2d_with_index
- prior_box
- reduce_sum
- reduce_mean
- reduce_max
- reshape
- reshape2
- flatten
- flatten2
- scale
- shape
- slice
- softmax
- split
- squeeze
- squeeze2
- swish
- transpose
- yolo_box

## 参考示例演示

### 准备设备环境
- 请确保您的 Sophon 加速卡已经可以在主机中正常工作。您可以对照下面的步骤快速验证，如有任何问题，请联系BITMAIN 解决。
    - 可以使用 BMNNSDK 内附带的 `bm-smi` 程序进行测试，如果设备已经正常驱动，您应该能看到一个(或多个)PCIE 模式的加速设备；
    - 可以使用 `ls /dev/bm*`, 您应该能看到若干个 `bm` 前缀的设备，例如 `/dev/bmdev-ctl /dev/bm-sophon0`。
- 简易安装指南(仅供参考，请以 BITMAIN 的安装指南为准)
  - 可以使用 `bmnnsdk_root/scripts/` 目录的 `install_libs.sh` 及 `sudo install_driver_pcie.sh` 完成安装；
  - 对于部分 BM1684 设备，为了启用动态编译功能，您可能需要为每个芯片启用 icache。在 `bmnnsdk_root/bin/x86` 目录下，使用 `./test_update_fw ./bm168x_bmdnn_en_icache.bin ./bm168x_bmdnn_s_en_icache.bin chip_id` 来启用 icache，`chip_id` 为 0，1，2 等值，代表不同芯片。

### 准备本地编译环境

- 目前仅在 Ubuntu 16.04 环境进行过测试，为了避免环境不一致带来的麻烦，建议使用 Docker 编译环境，请先根据[编译环境准备](../source_compile/compile_env)下载好 `paddlepaddle/paddle-lite` Docker 镜像。
- 在执行 `docker run` 启动容器时，请确保宿主机内 `/dev/bm*` 设备均被正确映射到容器中。可以参考下面的指令启动容器。
  
```bash
sudo docker run -it \
     --name work_bm \
     -v $HOME:/code \
     --device=/dev/bm1682-dev0:/dev/bm1682-dev0 \
     --device=/dev/bmdev-ctl:/dev/bmdev-ctl \
     --net=host \
     paddlepaddle/paddle-lite:latest /bin/bash

```


### 编译 Paddle Lite 工程
1. 下载代码
  
```bash
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
```

2. 编译

```bash
# 进入代码目录
cd Paddle-Lite

# 运行编译脚本
./lite/tools/build_bm.sh --target_name=BM1682
# 或 ./lite/tools/build_bm.sh --target_name=BM1684

```

3. 编译结果说明
编译产物将输出至 `build.lite.bm/inference_lite_lib` 目录下,该目录结构为
```bash
.
|-- cxx 
|   |-- include #cxx 头文件目录
|   `-- lib #cxx 库目录
|       `-- third_party #第三方 cxx 依赖库目录
`-- demo
    `-- cxx # cxx 示例程序

```

### 运行 demo

```bash
# 测试 demo 在 build.lite.bm/inference_lite_lib/demo/cxx/ 目录下
cd build.lite.bm/inference_lite_lib/demo/cxx/

wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz

tar -xvf mobilenet_v1.tar.gz

./build.sh

./mobilenet_full_api ./mobilenet_v1 224 224

# 如果运行正常,程序最后会输出 Done.
```
