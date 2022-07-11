# 芯原 TIM-VX 部署示例

Paddle Lite 已支持通过 TIM-VX 的方式调用芯原 NPU 算力的预测部署。
其接入原理是与其他接入 Paddle Lite 的新硬件类似，即加载并分析 Paddle 模型，首先将 Paddle 算子转成 NNAdapter 标准算子，其次再通过 TIM-VX 的组网 API 进行网络构建，在线编译模型并执行模型。

需要注意的是，芯原（verisilicon）作为 IP 设计厂商，本身并不提供实体SoC产品，而是授权其 IP 给芯片厂商，如：晶晨（Amlogic），瑞芯微（Rockchip）等。因此本文是适用于被芯原授权了 NPU IP 的芯片产品。只要芯片产品没有大副修改芯原的底层库，则该芯片就可以使用本文档作为 Paddle Lite 推理部署的参考和教程。在本文中，晶晨 SoC 中的 NPU 和 瑞芯微 SoC 中的 NPU 统称为芯原 NPU。

本文档与[ 晶晨 NPU 部署示例 ](./amlogic_npu)和[ 瑞芯微 NPU 部署示例 ](./rockchip_npu)中所描述的部署示例相比，虽然涉及的部分芯片产品相同，但前者是通过 IP 厂商芯原的 TIM-VX 框架接入 Paddle Lite，后二者是通过各自芯片 DDK 接入 Paddle Lite。接入方式不同，支持的算子和模型范围也有所区别。TIM-VX 支持的算子和模型种类更多。

## 支持现状

### 已支持的芯片

- Amlogic A311D

- Amlogic S905D3

- Amlogic C308X

- Rockchip RV1109

- Rockchip RV1126

- Rockchip RK1808

- NXP i.MX 8M Plus

  注意：理论上支持所有经过芯原授权了 NPU IP 的 SoC（须有匹配版本的 NPU 驱动，下文描述），上述为经过测试的部分芯片型号。

### 已支持的 Paddle 模型

#### 模型

- [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)
- [resnet50_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/resnet50_int8_224_per_layer.tar.gz)
- [ssd_mobilenet_v1_relu_voc_int8_300_per_layer](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_int8_300_per_layer.tar.gz)

#### 性能

- 测试环境
  - 编译环境
    - Ubuntu 16.04，GCC 5.4 for ARMLinux armhf and aarch64

  - 硬件环境
    - Amlogic A311D
      - CPU：4 x ARM Cortex-A73 + 2 x ARM Cortex-A53
      - NPU：5 TOPs for INT8
    - Amlogic S905D3
      - CPU：2 x ARM Cortex-55
      - NPU：1.2 TOPs for INT8
    - Amlogic C308X
      - CPU：2 x ARM Cortex-55
      - NPU：4 TOPs for INT8
    - Rockchip RK1808
      - CPU：2 x ARM Cortex-35
      - NPU：3 TOPs for INT8
    - Rockchip RV1109
      - CPU：2 x ARM Cortex-A7
      - NPU：1.2 TOPs for INT8
    - Rockchip RV1126
      - CPU：4 x ARM Cortex-A7
      - NPU：2 TOPs for INT8
    - NXP i.MX 8M Plus
      - CPU：4 x ARM Cortex-53
      - NPU：5 TOPs for INT8

- 测试方法
  - warmup=1, repeats=5，统计平均时间，单位是 ms
  - 线程数为1，`paddle::lite_api::PowerMode CPU_POWER_MODE`设置为` paddle::lite_api::PowerMode::LITE_POWER_HIGH `
  - 分类模型的输入图像维度是{1, 3, 224, 224}
  
- 测试结果

  |模型 |A311D||S905D3||C308X||RK1808||RV1109||RV1126||i.MX 8M Plus||
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  |  |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |
  |mobilenet_v1_int8_224_per_layer| 81.63213 | 5.1125| 280.4659| 12.8081 |167.623|6.9828|264.6235|6.139|335.0399|6.1995|281.63 | 5.1120 |106.656 | 3.212360|
  |resnet50_int8_224_per_layer| 390.4983| 17.5832 | 787.5323 | 41.3139|949.5|32.354 |1188.3469|18.1989|1660.2725|24.8895|590.8854| 20.5832|409.325| 12.6551 |
  |ssd_mobilenet_v1_relu_voc_int8_300_per_layer| 134.9915| 15.2167 | 295.4891| 40.1089 |196.377|26.8084|542.56|16.84|512.101|22.187|261.5986| 20.12287|159.3365| 14.2235 |

### 已支持（或部分支持）NNAdapter 的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

### 准备设备环境

- 确定开发板 NPU 驱动版本
  - 由于晶晨 SoC、瑞芯微1代 Soc、恩智浦 i.MX 8M Plus 等 使用芯原 NPU IP，因此，部署前要保证芯原 Linux Kernel NPU 驱动—— galcore.so 版本及所适用的芯片型号与依赖库保持一致。
  - 请登录开发板，并通过命令行输入 `dmesg | grep Galcore` 查询 NPU 驱动版本。
    - 请务必注意，建议 NPU 驱动版本为：

  |SoC 厂家|驱动板本|
  |---|---|
  |Amlogic|6.4.4.3|
  |Rockchip|6.4.3.5|
  |NXP|6.4.3.p1|

    - 举个例子，以晶晨 Amlogic A311D 为例，需要为 6.4.4.3（其他搭载了芯原 NPU 的 SoC 驱动版本要求参照上表）：
      ```shell
      $ dmesg | grep Galcore
      [   24.140820] Galcore version 6.4.4.3.310723AAA
      ```
    - 如果当前版本就符合上述 ，直接跳过本环节。
  - 如果当前版本不符合上述，请用户仔细阅读以下内容，以保证底层 NPU 驱动环境正确。
  - 前提科普：由于使用芯原 NPU 的 SoC 众多，且同一 SoC 芯片、不同开发板之间存在 linux kernel 版本的差异，该差异直接影响 NPU 驱动的通用性。用户可以如此理解：1）芯原提供 NPU IP、2）芯片商（晶晨、瑞芯微、恩智浦）提供 SoC 芯片、3）开发板商（产品商）提供实际开发板产品，这三方共同形成了芯片产业生态。
  - 芯原 NPU 的开发板主要有两个关键环境因素务必对齐：1）galcore.ko（既 NPU 驱动文件）；2）NPU 依赖库。上述两者需要版本对齐，缺一不可。
  - 有两种方式可以修改当前的 NPU 驱动版本及其依赖库：
    - 『方法 1』：手动替换 NPU 驱动版本及其依赖库。（**推荐**）
    - 『方法 2』：刷机，刷取 NPU 驱动版本符合要求的固件。
  - 我们首先描述『方法 1』手动替换驱动文件和依赖库，先行下载并解压[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，其中包含不同版本、不同芯片型号的 galcore.ko（既 NPU 驱动文件）和 NPU 依赖库。
    - 下表会罗列部分市面常见开发板的情况，以及我们在 [PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz) 中提供的现成的驱动文件和依赖库。请照着下表格，找到自己手中对应设备的芯片、开发板、Linux Kernel 版本（可命令行输入 uname -a 查看），从而获取到真正需要的 1）galcore.ko（既 NPU 驱动文件）；2）NPU 依赖库。并且分别将 galcore.ko 上传至开发板后，insmod galcore.ko，以及输入表格中的命令刷取正确的NPU 依赖库（软链接）。更加详细易懂的使用步骤会在下表格后描述。

|SoC 型号 | 开发板厂家 |开发板型号|OS |推荐Linux Kernl 版本|推荐NPU驱动版本 |是否提供galcore.ko驱动文件 |galcore.ko驱动文件路径 |是否提供 NPU 依赖库|刷取 NPU 依赖库软链接命令|
|---|---|---|---|---|---|---|---|---|---|
|Amlogic A311D |世野科技 Khadas |VIM3 [购买链接](https://www.khadas.cn/product-page/vim3)|android |4.9.113 |6.4.4.3 |是 |PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/a311d/4.9.113|是|cd PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 a311d|
|Amlogic A311D |世野科技 Khadas |VIM3 [购买链接](https://www.khadas.cn/product-page/vim3)|linux |4.9.241 |6.4.4.3 |是 |PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/a311d/4.9.241|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 a311d|
|Amlogic A311D |荣品 |PR-A311D [购买链接](https://item.taobao.com/item.htm?spm=a1z10.1-c-s.w4004-23440679120.23.849147calaBS8s&id=614553849827)|linux | 4.9.113|6.4.4.3|是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/a311d/4.9.113|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 a311d|
|Amlogic A311D |*其他* || linux | 4.9.113|6.4.4.3|是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/a311d/4.9.113|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 a311d|
|Amlogic 905D3 |世野科技 Khadas |VIM3L [购买链接](https://www.khadas.cn/product-page/vim3l)|android|4.9.113 | 6.4.4.3| 是|PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/s905d3/4.9.113| 是|cd PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 s905d3|
|Amlogic 905D3 |世野科技 Khadas |VIM3L [购买链接](https://www.khadas.cn/product-page/vim3l)|linux|4.9.241 | 6.4.4.3| 是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/s905d3/4.9.241| 是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 s905d3|
|Amlogic 905D3 |荣品|RP-S905 [购买链接](https://item.taobao.com/item.htm?spm=a1z10.5-c-s.w4002-22747001949.12.6c774cb50jm33m&id=615270715583)|linux|4.9.113 | 6.4.4.3| 是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/s905d3/4.9.113| 是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 s905d3|
|Amlogic 905D3 |*其他*||linux|4.9.113 | 6.4.4.3| 是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/s905d3/4.9.113| 是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_4_3 s905d3|
|Amlogic C308X |*其他*||linux|4.19.81 | 6.4.4.3| 是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_4_3/lib/c308x/4.19.81|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon__timvx && ./switch_viv_sdk.sh 6_4_4_3 c308x|
|Rockchip RV1109|瑞芯微|RV1109 DDR3 EVB|linux|4.19.111|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/viv_sdk_6_4_3_5/1109/4.19.111|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 1109|
|Rockchip RV1109|荣品|RP-RV1109 [购买链接](https://item.taobao.com/item.htm?spm=a1z10.3-c-s.w4002-22747001942.9.143851d6bSPUFo&id=633366427160)| linux|4.19.111|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/viv_sdk_6_4_3_5/1109/4.19.111|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 1109|
|Rockchip RV1109|*其他*||linux|4.19.111|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/viv_sdk_6_4_3_5/1109/4.19.111|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 1109|
|Rockchip RV1126|瑞芯微|RV1126 DDR3 EVB|linux|4.19.111|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/viv_sdk_6_4_3_5/1126/4.19.111|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 1126|
|Rockchip RV1126|荣品|RP-RV1126 [购买链接](https://item.taobao.com/item.htm?spm=a1z10.1-c-s.w4004-23440679120.6.849147calaBS8s&id=641752963533&mt=)|linux|4.19.111|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/viv_sdk_6_4_3_5/1126/4.19.111|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 1126|
|Rockchip RV1126|*其他*||linux|4.19.111|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/viv_sdk_6_4_3_5/1126/4.19.111|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 1126|
|Rockchip RK1808|瑞芯微|RK1808 DDR3 EVB|linux|4.4.194|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_3_5/lib/rk1808/4.4.194|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 rk1808|
|Rockchip RK1808|荣品|RP-RK1808 [购买链接](https://item.taobao.com/item.htm?spm=a1z10.5-c-s.w4002-22747001949.11.57821a67E535sj&id=615447675327)|linux|4.4.194|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_3_5/lib/rk1808/4.4.194|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 rk1808|
|Rockchip RK1808|*其他*||linux|4.4.194|6.4.3.5|是|PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/viv_sdk_6_4_3_5/lib/rk1808/4.4.194|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_5 rk1808|
|NPX i.MX 8M Plus|*其他*||linux|5.4.70|6.4.3.p1|否|目前常见的 NPX i.MX 8M Plus 开发板的系统较为特殊，其驱动文件是 buildin 在系统中的|是|cd PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx && ./switch_viv_sdk.sh 6_4_3_p1 imx8mp|

    - 详细步骤：
      - 第一步：在上表格中，根据芯片型号、开发板商，找到对应自己的开发板那一行。
      - 第二步：登录开发板，命令行输入 uname -a 来确定自己开发板的 Linux Kernel 是否和表格中一致，如果不一致，请跳转至『方法 2』.
      - 第三步：在表格里找到对应行中 galcore.ko 文件的路径，将 galcore.ko 其上传至开发板。
      - 第四步：登录开发板，命令行输入 `sudo rmmod galcore` 来卸载原始驱动，输入 `sudo insmod galcore.ko` 来加载传上设备的驱动。（是否需要 sudo 根据开发板实际情况，部分 adb 链接的设备请提前 adb root）。此步骤如果操作失败，请跳转至『方法 2』.
      - 第五步：在开发板中输入 `dmesg | grep Galcore` 查询 NPU 驱动版本，确定为：晶晨6.4.4.3，瑞芯微6.4.3.5，NXP 6.4.3.p1。
      - 第六步：在表格里找到对应设备行的最后一列，在下载了[PaddleLite-generic-demo](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)的PC目录下输入表中命令，切换成对应的 NPU 依赖库软链接。
      - 至此，前期的环境准备就已经完成，恭喜您，可以完美复现我们需要的环境。
      - 最后，所有开发板都有开机默认加载路径，建议用户把之前上传的 galcore.ko 文件放在开发板的系统默认加载目录下（一般情况为 XXX/lib/modules/ 下，用户可以在开发板的 / 目录下 `find -name galcore.ko` 来得知应该放在哪里），如此下次开机便能自动加载我们需要的 NPU 驱动。

  - 如果上述方法在过程中失败，那我们使用『方法 2』刷机：
    - 根据具体的开发板型号，向开发板卖家或官网客服索要对应上表中版本 NPU 驱动对应的固件和刷机方法。
      - 在此额外提供 khadas 开发板 VIM3|VIM3L 的 6.4.4.3 固件以及官方教程链接：
        - 刷机镜像（包含 NPU 驱动文件和芯原相关依赖库，分别提供 khadas 官方服务器下载地址，和飞桨服务器的下载地址，均可下载使用）：
          - VIM3 Android：VIM3_Pie_V210908：[官方链接](https://dl.khadas.com/Firmware/VIM3/Android/VIM3_Pie_V210908.7z)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3/VIM3_Pie_V210908.7z)
          - VIM3 Linux：VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625：[官方链接](http://dl.khadas.com/firmware/VIM3/Ubuntu/EMMC/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)
          - VIM3L Android：VIM3L_Pie_V210906：[官方链接](https://dl.khadas.com/Firmware/VIM3L/Android/VIM3L_Pie_V210906.7z)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3l/VIM3L_Pie_V210906.7z)
          - VIM3L Linux：VIM3L_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625：[官方链接](https://dl.khadas.com/Firmware/VIM3L/Ubuntu/EMMC/VIM3L_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3l/VIM3L_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)
        - 官方刷机教程：[VIM3/3L Android 文档](https://docs.khadas.com/android/zh-cn/vim3/) , [VIM3/3L Linux 文档](https://docs.khadas.com/linux/zh-cn/vim3)，其中有详细描述刷机方法。
      - 其余开发板用户可向开发板卖家或官网客服索要 晶晨6.4.4.3，瑞芯微6.4.3.5，NXP 6.4.3.p1 版本的 NPU 驱动对应的固件和刷机方法。

- 示例程序和 Paddle Lite 库的编译建议采用交叉编译方式，通过 `adb`或`ssh` 进行设备的交互和示例程序的运行。


### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[ Docker 环境准备](../source_compile/docker_environment)中的 Docker 开发环境进行配置；
- 由于有些设备只提供网络访问方式（根据开发版的实际情况），需要通过 `scp` 和 `ssh` 命令将交叉编译生成的Paddle Lite 库和示例程序传输到设备上执行，因此，在进入 Docker 容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下（注意其中软链接为 switch_viv_sdk.sh 根据芯片型号和 NPU 驱动版本创建依赖库的软链接）：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - images
            - tabby_cat.jpg # 测试图片
            - tabby_cat.raw # 经过 convert_to_raw_image.py 处理后的 RGB Raw 图像
          - labels
            - synset_words.txt # 1000 分类 label 文件
          - models
            - mobilenet_v1_int8_224_per_layer
              - __model__ # Paddle fluid 模型组网文件，可使用 netron 查看网络结构
              — conv1_weights # Paddle fluid 模型参数文件
              - batch_norm_0.tmp_2.quant_dequant.scale # Paddle fluid 模型量化参数文件
              — subgraph_partition_config_file.txt # 自定义子图分割配置文件
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.arm64 # arm64 编译工作目录
            - image_classification_demo # 已编译好的，适用于 arm64 的示例程序
          - build.linux.armhf # armhf编译工作目录
            - image_classification_demo # 已编译好的，适用于 armhf 的示例程序
          - build.android.armeabi-v7a # Android armv7编译工作目录
            - image_classification_demo # 已编译好的，适用于 Android armv7 的示例程序
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序ssh运行脚本
          - run_with_adb.sh # 示例程序adb运行脚本
      - libs
        - PaddleLite
          - linux
            - arm64 # Linux 64 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - verisilicon_timvx # 芯原 DDK、NNAdapter 运行时库、device HAL 库
                  - libArchModelSw.so -> ./viv_sdk_6_4_4_3/lib/libArchModelSw.so
                  - libCLC.so -> ./viv_sdk_6_4_4_3/lib/libCLC.so
                  - libGAL.so -> ./viv_sdk_6_4_4_3/lib/libGAL.so
                  - libNNArchPerf.so -> ./viv_sdk_6_4_4_3/lib/libNNArchPerf.so
                  - libNNGPUBinary.so -> ./viv_sdk_6_4_4_3/lib/a311d/libNNGPUBinary.so
                  - libNNVXCBinary.so -> ./viv_sdk_6_4_4_3/lib/a311d/libNNVXCBinary.so
                  - libOpenCL.so -> ./viv_sdk_6_4_4_3/lib/libOpenCL.so
                  - libOpenVX.so -> ./viv_sdk_6_4_4_3/lib/libOpenVX.so
                  - libOpenVXU.so -> ./viv_sdk_6_4_4_3/lib/libOpenVXU.so
                  - libOvx12VXCBinary.so -> ./viv_sdk_6_4_4_3/lib/a311d/libOvx12VXCBinary.so
                  - libVSC.so -> ./viv_sdk_6_4_4_3/lib/libVSC.so
                  - libverisilicon_timvx.so # NNAdapter device HAL 库
                  - libnnadapter.so  # NNAdapter 运行时库
                  - libtim-vx.so # 芯原 TIM-VX 库
                  - switch_viv_sdk.sh # 根据芯片型号和 NPU 驱动版本创建依赖库的软链接
                  - viv_sdk_6_4_4_3
                    - include
                    - lib
                    - a311d # 针对 a311d 平台
                      - 4.9.241
                        - galcore.ko # NPU 驱动文件
                      - libNNGPUBinary.so # 芯原 DDK
                      - libNNVXCBinary.so # 芯原 DDK
                      - libOvx12VXCBinary.so # 芯原 DDK
                    - libArchModelSw.so # 芯原 DDK
                    - libCLC.so # 芯原 DDK
                    - libGAL.so # 芯原 DDK
                    - libNNArchPerf.so # 芯原 DDK
                    - libOpenCL.so # 芯原 DDK
                    - libOpenVX.so # 芯原 DDK
                    - libOpenVXU.so # 芯原 DDK
                    - libVSC.so # 芯原 DDK
                    - libovxlib.so
                    - s905d3 # 针对 s905d3 平台
                      - 4.9.241
                        - galcore.ko
                      ...
                    - c308x # 针对 c308x 平台
                      - 4.19.81
                        - galcore.ko
                      ...
                  - viv_sdk_6_4_3_5
                    - rk1808 # 针对 rk1808 平台
                      - 4.4.194
                        - galcore.ko
                      ...
                  - viv_sdk_6_4_4_p1
                    - imx8mp # 针对 nxp i.MX 8M Plus 平台
                      ...
                - libpaddle_full_api_shared.so # 预编译 PaddleLite full api 库
                - libpaddle_light_api_shared.so # 预编译 PaddleLite light api 库
            - armhf # Linux 32 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - verisilicon_timvx # 芯原 DDK、NNAdapter 运行时库、device HAL 库
                  - viv_sdk_6_4_3_5
                    - 1109 # 针对 rv1109 平台
                      - 4.19.111
                        - galcore.ko
                      ...
                    - 1126 # 针对 rv1126平台
                      - 4.19.111
                        - galcore.ko
                      ...
                  ...
                ...
          - android
           - armeabi-v7a # Android 32 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - verisilicon_timvx # 芯原 DDK、NNAdapter 运行时库、device HAL 库
                  - libCLC.so -> ./viv_sdk_6_4_4_3/lib/libCLC.so
                  - libGAL.so -> ./viv_sdk_6_4_4_3/lib/libGAL.so
                  - libNNArchPerf.so -> ./viv_sdk_6_4_4_3/lib/libNNArchPerf.so
                  - libNNGPUBinary.so -> ./viv_sdk_6_4_4_3/lib/s905d3/libNNGPUBinary.so
                  - libNNVXCBinary.so -> ./viv_sdk_6_4_4_3/lib/s905d3/libNNVXCBinary.so
                  - libOpenCL.so -> ./viv_sdk_6_4_4_3/lib/libOpenCL.so
                  - libOpenVX.so -> ./viv_sdk_6_4_4_3/lib/libOpenVX.so
                  - libOpenVXU.so -> ./viv_sdk_6_4_4_3/lib/libOpenVXU.so
                  - libOvx12VXCBinary.so -> ./viv_sdk_6_4_4_3/lib/s905d3/libOvx12VXCBinary.so
                  - libVSC.so -> ./viv_sdk_6_4_4_3/lib/libVSC.so
                  - libverisilicon_timvx.so # NNAdapter device HAL 库
                  - libarchmodelSw.so -> ./viv_sdk_6_4_4_3/lib/libarchmodelSw.so
                  - libnnadapter.so # NNAdapter 运行时库
                  - libtim-vx.so # 芯原 TIM-VX 库
                  - switch_viv_sdk.sh # 根据芯片型号和 NPU 驱动版本创建依赖库的软链接
                  - viv_sdk_6_4_4_3
                    - include
                    - lib
                      - a311d # 针对 a311d 平台
                        - 4.9.113
                          - VERSION
                          - galcore.ko # NPU驱动
                        - libNNGPUBinary.so
                        - libNNVXCBinary.so
                        - libOvx12VXCBinary.so
                      - libCLC.so # 芯原 DDK
                      - libGAL.so # 芯原 DDK
                      - libNNArchPerf.so # 芯原 DDK
                      - libOpenCL.so 
                      - libOpenVX.so # 芯原 DDK
                      - libOpenVXU.so # 芯原 DDK
                      - libVSC.so # 芯原 DDK
                      - libarchmodelSw.so # 芯原 DDK
                      - s905d3 # 针对 s905d3 平台
                        - 4.9.113
                            - VERSION
                            - galcore.ko # NPU驱动
                        - libNNGPUBinary.so # 芯原 DDK
                        - libNNVXCBinary.so # 芯原 DDK
                        - libOvx12VXCBinary.so # 芯原 DDK
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
  ```

- 按照以下命令分别运行转换后的ARM CPU模型和 芯原 TIM-VX 模型，比较它们的性能和结果；

  ```shell
  注意：
  1）`run_with_adb.sh` 不能在 Docker 环境执行，否则可能无法找到设备，也不能在设备上运行。
  2）`run_with_ssh.sh` 不能在设备上运行，且执行前需要配置目标设备的 IP 地址、SSH 账号和密码。
  3）`build.sh` 根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
  4）`run_with_adb.sh` 入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
  5）`run_with_ssh.sh` 入参包括模型名称、操作系统、体系结构、目标设备、ip地址、用户名、用户密码等，需查阅注释信息配置正确的参数值。
  6）下述命令行示例中涉及的具体IP、SSH账号密码、设备序列号等均为示例环境，请用户根据自身实际设备环境修改。
  
  在 ARM CPU 上运行 mobilenet_v1_int8_224_per_layer 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  
  For SSH 连接开发板的使用场景
  #Linux arm64 命令：
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu IP地址 22 用户名 密码
  #Linux arm32 命令：
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux armhf cpu IP地址 22 用户名 密码
  #Android armeabi-v7a 命令：
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a cpu IP地址 22 用户名 密码
    (如下以 A311D(Linux 版) 为例，其他 SoC 也一样，仅性能有区别)
    warmup: 1 repeat: 15, average: 81.678067 ms, max: 81.945999 ms, min: 81.591003 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 1.352000 ms
    Prediction time: 81.678067 ms
    Postprocess time: 0.407000 ms
  
  For ADB 连接开发板的使用场景
  #Linux arm64 命令：
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu adb设备号
  #Linux arm32 命令：
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux armhf cpu adb设备号
  #Android armeabi-v7a 命令：
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a cpu adb设备号
    (如下以 S905D3(Android版) 为例，其他 SoC 也一样，仅性能有区别)
    warmup: 1 repeat: 5, average: 280.465997 ms, max: 358.815002 ms, min: 268.549812 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.199000 ms
    Prediction time: 280.465997 ms
    Postprocess time: 0.596000 ms
  
  ------------------------------
  
  在 芯原 NPU 上运行 mobilenet_v1_int8_224_per_layer 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  
  For SSH 连接开发板的使用场景
  #Linux arm64 命令：
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 verisilicon_timvx IP地址 22 用户名 密码
  #Linux arm32 命令：
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux armhf verisilicon_timvx IP地址 22 用户名 密码
  #Android armeabi-v7a 命令：
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a verisilicon_timvx IP地址 22 用户名 密码
    (如下以 A311D(Linux 版) 为例，其他 SoC 也一样，仅性能有区别，精度可能有细微差异)
    warmup: 1 repeat: 15, average: 5.112500 ms, max: 5.223000 ms, min: 5.009130 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 1.356000 ms
    Prediction time: 5.112500 ms
    Postprocess time: 0.411000 ms
  
  For ADB 连接开发板的使用场景
  #Linux arm64 命令：
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 verisilicon_timvx adb设备号
  #Linux arm32 命令：
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux armhf verisilicon_timvx adb设备号
  #Android armeabi-v7a 命令：
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a verisilicon_timvx adb设备号
    (如下以 S905D3(Android版) 为例，其他 SoC 也一样，仅性能有区别，精度可能有细微差异)
    warmup: 1 repeat: 5, average: 13.4116 ms, max: 14.7615 ms, min: 12.80810 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 3.170000 ms
    Prediction time: 13.4116 ms
    Postprocess time: 0.634000 ms
  ```
  
- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/images` 目录下，然后调用 `convert_to_raw_image.py` 生成相应的 RGB Raw 图像，最后修改 `run_with_adb.sh`、`run_with_ssh.sh` 的 IMAGE_NAME 变量即可；
- 重新编译示例程序：  
  ```shell
  注意：
  1）请根据 `buid.sh` 配置正确的参数值。
  2）需在 Docker 环境中编译。
  
  # 对于 Liunx 64位 系统
  ./build.sh linux arm64
  # 对于 Liunx 32位 系统
  ./build.sh linux armhf
  # 对于 Android 系统
  ./build.sh android armeabi-v7a
  ```

### 更新模型
- 通过 Paddle 训练或 X2Paddle 转换得到 MobileNetv1 foat32 模型[ mobilenet_v1_fp32_224 ](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)
- 通过 Paddle+PaddleSlim 后量化方式，生成[ mobilenet_v1_int8_224_per_layer 量化模型](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)
- 下载[ PaddleSlim-quant-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/tools/PaddleSlim-quant-demo.tar.gz)，解压后清单如下：
    ```shell
    - PaddleSlim-quant-demo
      - image_classification_demo
        - quant_post # 后量化
          - quant_post_rockchip_npu.sh # 一键量化脚本，Amlogic 和瑞芯微底层都使用芯原的 NPU，所以通用
          - README.md # 环境配置说明，涉及 PaddlePaddle、PaddleSlim 的版本选择、编译和安装步骤
          - datasets # 量化所需要的校准数据集合
            - ILSVRC2012_val_100 # 从 ImageNet2012 验证集挑选的 100 张图片
          - inputs # 待量化的 fp32 模型
            - mobilenet_v1
            - resnet50
          - outputs # 产出的全量化模型
          - scripts # 后量化内置脚本
    ```
- 查看 `README.md` 完成 PaddlePaddle 和 PaddleSlim 的安装
- 直接执行 `./quant_post_rockchip_npu.sh` 即可在 `outputs` 目录下生成mobilenet_v1_int8_224_per_layer 量化模型
  ```shell
  -----------  Configuration Arguments -----------
  activation_bits: 8
  activation_quantize_type: moving_average_abs_max
  algo: KL
  batch_nums: 10
  batch_size: 10
  data_dir: ../dataset/ILSVRC2012_val_100
  is_full_quantize: 1
  is_use_cache_file: 0
  model_path: ../models/mobilenet_v1
  optimize_model: 1
  output_path: ../outputs/mobilenet_v1
  quantizable_op_type: conv2d,depthwise_conv2d,mul
  use_gpu: 0
  use_slim: 1
  weight_bits: 8
  weight_quantize_type: abs_max
  ------------------------------------------------
  quantizable_op_type:['conv2d', 'depthwise_conv2d', 'mul']
  2021-08-30 05:52:10,048-INFO: Load model and set data loader ...
  2021-08-30 05:52:10,129-INFO: Optimize FP32 model ...
  I0830 05:52:10.139564 14447 graph_pattern_detector.cc:91] ---  detected 14 subgraphs
  I0830 05:52:10.148236 14447 graph_pattern_detector.cc:91] ---  detected 13 subgraphs
  2021-08-30 05:52:10,167-INFO: Collect quantized variable names ...
  2021-08-30 05:52:10,168-WARNING: feed is not supported for quantization.
  2021-08-30 05:52:10,169-WARNING: fetch is not supported for quantization.
  2021-08-30 05:52:10,170-INFO: Preparation stage ...
  2021-08-30 05:52:11,853-INFO: Run batch: 0
  2021-08-30 05:52:16,963-INFO: Run batch: 5
  2021-08-30 05:52:21,037-INFO: Finish preparation stage, all batch:10
  2021-08-30 05:52:21,048-INFO: Sampling stage ...
  2021-08-30 05:52:31,800-INFO: Run batch: 0
  2021-08-30 05:53:23,443-INFO: Run batch: 5
  2021-08-30 05:54:03,773-INFO: Finish sampling stage, all batch: 10
  2021-08-30 05:54:03,774-INFO: Calculate KL threshold ...
  2021-08-30 05:54:28,580-INFO: Update the program ...
  2021-08-30 05:54:29,194-INFO: The quantized model is saved in ../outputs/mobilenet_v1
  post training quantization finish, and it takes 139.42292165756226.
  
  -----------  Configuration Arguments -----------
  batch_size: 20
  class_dim: 1000
  data_dir: ../dataset/ILSVRC2012_val_100
  image_shape: 3,224,224
  inference_model: ../outputs/mobilenet_v1
  input_img_save_path: ./img_txt
  save_input_img: False
  test_samples: -1
  use_gpu: 0
  ------------------------------------------------
  Testbatch 0, acc1 0.8, acc5 1.0, time 1.63 sec
  End test: test_acc1 0.76, test_acc5 0.92
  --------finish eval int8 model: mobilenet_v1-------------
  ```
  - 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成 TIM-VX 模型，仅需要将 `valid_targets` 设置为 `verisilicon_timvx`, `arm` 即可。
  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=verisilicon_timvx,arm
  ```
### 更新支持 TIM-VX 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  # 注意：编译中依赖的 verisilicon_timvx 相关代码和依赖项会在后续编译脚本中自动下载，无需用户手动下载。
  ```
  
- 编译并生成 `Paddle Lite+Verisilicon_TIMVX` 的部署库

  - For A311D(Linux 版) & S905D3(Linux 版) & C308X(Linux 版)
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_4_3_generic.tgz
      
      ```
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_4_3_generic.tgz full_publish
      
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libverisilicon_timvx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 芯原 TIM-VX 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libtim-vx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```
  
  - For A311D(Android 版) &S905D3(Android 版)
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_exception=ON --with_cv=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_android_9_armeabi_v7a_6_4_4_3_generic.tgz
      ```
  
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_exception=ON --with_cv=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_android_9_armeabi_v7a_6_4_4_3_generic.tgz full_publish
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libverisilicon_timvx.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/
      # 替换 芯原 TIM-VX 库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libtim-vx.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      # 替换 libpaddle_full_api_shared.so(仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```
  
  - For RV1109 & RV1126
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_cv=ON --with_exception=ON --arch=armv7hf  --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm32_6_4_3_5_generic.tgz
      
      ```
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_cv=ON --with_exception=ON --arch=armv7hf  --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm32_6_4_3_5_generic.tgz full_publish
      
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libverisilicon_timvx.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/
      # 替换 芯原 TIM-VX 库
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libtim-vx.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/verisilicon_timvx/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      ```
  - For RK1808
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_3_5_generic.tgz
      
      ```
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_3_5_generic.tgz full_publish
      
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libverisilicon_timvx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 芯原 TIM-VX 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libtim-vx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```
  - For NXP imx8m plus
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_3_p1_generic.tgz
      
      ```
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_3_p1_generic.tgz full_publish
      
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libverisilicon_timvx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 芯原 TIM-VX 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libtim-vx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```
- 替换头文件后需要重新编译示例程序

## 其它说明

- Paddle Lite 研发团队正在持续扩展基于TIM-VX的算子和模型。
