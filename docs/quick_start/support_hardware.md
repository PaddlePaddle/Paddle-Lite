
# 硬件支持及示例


## ARM CPU
Paddle Lite支持[ARM Cortex-A系列处理器](https://en.wikipedia.org/wiki/ARM_Cortex-A)，支持列表如下:
### 32bit(ARMv7a)
- Cortex-A5
- Cortex-A7
- Cortex-A8
- Cortex-A9
- Cortex-A12
- Cortex-A15
- Cortex-A17(RK3288)
- Cortex-A32
### 64bit(ARMv7a, ARMv8a)
- Cortex-A35
- Cortex-A53(树莓派3)
- Cortex-A55
- Cortex-A57(Nvidia tx1，Nvidia tx2， 高通810等)
- Cortex-A72(麒麟95X，高通820, RK3399，树莓派4等)
- Cortex-A73(麒麟960，麒麟970，高通835, 联发科X30等)
- Cortex-A75(高通845等)
- Cortex-A76(麒麟980，麒麟990，高通855，高通730，联发科G90等）
- Cortex-A77
- ARMv8-A compatible(Apple A系列处理器, Nvidia tegra, Qualcomm Kryo, Falkor, Samsung Mongoose)
### ARM CPU 环境准备及示例程序
- Android：[Android 工程示例](../demo_guides/android_app_demo)
- iOS：[iOS 工程示例](../demo_guides/ios_app_demo)
- Linux(ARM)：[Linux(ARM) 工程示例](../demo_guides/linux_arm_demo)

## X86 CPU
Paddle Lite当前支持所有同时支持AVX及FMA指令集的X86 CPU，正在完善SSE指令集的实现。因为X86设备众多，不一一列举。
- 下面仅列举判断当前CPU是否同时支持AVX及FMA指令集（判断Paddle Lite是否支持该CPU）的方法：
  - 目标设备是linux：执行命令`cat /proc/cpuinfo`查看
  - 目标设备是windows：利用免费工具`CPU-Z`查看
### X86 CPU 环境准备及示例程序
- [PaddleLite使用X86预测部署](../demo_guides/x86)

## 移动端GPU
Paddle Lite支持多种移动端GPU，包括ARM Mali、Qualcomm Adreno、Apple A Series、Nvidia Tegra等系列GPU设备，支持列表如下：
- ARM Mali G 系列
- Qualcomm Adreno 系列
- Apple A 系列
- Nvida Tegra系列: Tegra X1, Tegra X2, Jetson Nano, Xavier
### 移动端GPU 环境准备及示例程序
- [PaddleLite使用OpenCL预测部署](../demo_guides/opencl)

## FPGA
Paddle Lite支持 **百度 FPGA**，支持列表如下：
- 百度Edgeboard系列：ZU9, ZU5, ZU3

Paddle Lite支持 **英特尔 (Intel) FPGA**，支持列表如下：
- 支持芯片：英特尔FPGA Cyclone V系列芯片
- 支持设备：
  - 海运捷讯C5MB（英特尔FPGA Cyclone V）开发板
  - 海运捷讯C5CB（英特尔FPGA Cyclone V）开发板
  - 海运捷讯C5TB（英特尔FPGA Cyclone V）开发板

### FPGA 环境准备及示例程序
- 百度 FPGA：[PaddleLite使用FPGA预测部署](../demo_guides/fpga)
- 英特尔 (Intel) FPGA：[PaddleLite使用英特尔FPGA预测部署](../demo_guides/intel_fpga)


## 百度 (Baidu) XPU
Paddle Lite支持百度XPU，支持列表如下：
- 百度昆仑818-100芯片
- 百度昆仑818-300芯片

### 百度 (Baidu) XPU 环境准备及示例程序
- [PaddleLite使用百度XPU预测部署](../demo_guides/baidu_xpu)

## 华为 (Huawei) 麒麟NPU
Paddle Lite支持华为达芬奇架构麒麟NPU，支持列表如下：
- 支持芯片：Kirin 810/990/985/9000
- 支持设备：
  * Kirin 9000：HUAWEI Mate 40pro系列
  * Kirin 9000E：HUAWEI Mate 40系列
  * Kirin 990 5G：HUAWEI Mate 30pro系列，P40pro系列
  * Kirin 990：HUAWEI Mate 30系列, 荣耀 V20系列, nova 6系列，P40系列，Mate Xs
  * Kirin 985：HUAWEI nova 7 5G，nova 7 Pro 5G，荣耀 30
  * Kirin 820：HUAWEI nova 7 SE 5G，荣耀 30S
  * Kirin 810：HUAWEI nova 5系列，nova 6 SE，荣耀 9X系列，荣耀 Play4T Pro

### 华为 (Huawei) 麒麟NPU 环境准备及示例程序
- [PaddleLite使用华为麒麟NPU预测部署](../demo_guides/huawei_kirin_npu)

## 华为 (Huawei) 昇腾NPU
Paddle Lite已支持华为昇腾NPU（Ascend310）在x86和Arm服务器上进行预测部署
- 支持设备：
  * Ascend 310：Atlas 300I推理卡（型号：3000/3010)
  * Atlas 200 DK开发者套件
  * Atlas 800推理服务器（型号：3000/3010）
### 华为 (Huawei) 昇腾NPU 环境准备及示例程序
- [PaddleLite使用华为昇腾NPU预测部署](../demo_guides/huawei_ascend_npu)

## 瑞芯微 (Rockchip) NPU
Paddle Lite支持 瑞芯微 (Rockchip) NPU，支持列表如下：
- 支持芯片：RK1808, RK1806，暂不支持RK3399Pro
- 支持设备：RK1808/1806 EVB，TB-RK1808S0
### 瑞芯微 (Rockchip) NPU 环境准备及示例程序
- [PaddleLite使用瑞芯微NPU预测部署](../demo_guides/rockchip_npu)

## 联发科 (MediaTek) APU
Paddle Lite支持 联发科 (MediaTek) APU，支持列表如下：
- 支持芯片：MT8168/MT8175，及其他智能芯片
- 支持设备：MT8168-P2V1 Tablej
### 联发科 (MediaTek) APU 环境准备及示例程序
- [PaddleLite使用联发科APU预测部署](../demo_guides/mediatek_apu)

## Amlogic NPU
Paddle Lite支持 Amlogic NPU, 支持列表如下：
- 支持芯片：C308X，A311D，S905D3(Android版本)
### Amlogic NPU 环境准备及示例程序
- [PaddleLite使用Amlogic NPU预测部署](../demo_guides/amlogic_npu)

## 颖脉 (Imagination) NNA
Paddle Lite支持 颖脉 (Imagination) NNA，支持列表如下：
- 支持芯片：紫光展锐虎贲T7510
- 支持设备：海信F50，Roc1开发板（基于T7510的微信电脑主板）
### 颖脉 (Imagination) NNA 环境准备及示例程序
- [PaddleLite使用颖脉NNA预测部署](../demo_guides/imagination_nna)

## 比特大陆（Bitmain）TPU
Paddle Lite支持 比特大陆（Bitmain）TPU，支持列表如下：
- 支持芯片：Sophon BM1682，Sophon BM1684
- 支持设备：
  * Sophon SC3 加速卡 (BM1682 X86 PCI-E)
  * Sophon SC5 加速卡 (BM1684 X86 PCI-E)
### 比特大陆（Bitmain）TPU 环境准备及示例程序
- [PaddleLite使用Bitmain：Sophon BM1682/BM1684 预测部署](../demo_guides/bitmain)
