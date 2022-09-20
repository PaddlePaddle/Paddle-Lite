
# 硬件支持及示例

## ARM CPU

Paddle Lite 支持 [ARM Cortex-A 系列处理器](https://en.wikipedia.org/wiki/ARM_Cortex-A)，支持列表如下:

### 32bit (ARMv7a)

- Cortex-A5
- Cortex-A7
- Cortex-A8
- Cortex-A9
- Cortex-A12
- Cortex-A15
- Cortex-A17 (RK3288)
- Cortex-A32

### 64bit (ARMv7a, ARMv8a)

- Cortex-A35
- Cortex-A53 (树莓派 3)
- Cortex-A55
- Cortex-A57 (Nvidia tx1，Nvidia tx2， 高通 810 等)
- Cortex-A72 (麒麟 95X，高通 820, RK3399，树莓派 4 等)
- Cortex-A73 (麒麟 960，麒麟 970，高通 835, 联发科 X30 等)
- Cortex-A75 (高通 845等)
- Cortex-A76 (麒麟 980，麒麟 990，高通 855，高通 730，联发科 G90 等）
- Cortex-A77
- ARMv8-A compatible (Apple A 系列处理器, Nvidia tegra, Qualcomm Kryo, Falkor, Samsung Mongoose)

### ARM CPU 环境准备及示例程序

- Android: [Android 工程示例](../demo_guides/android_app_demo)
- IOS: [IOS 工程示例](../demo_guides/ios_app_demo)
- Linux(ARM): [Linux(ARM) 工程示例](../demo_guides/linux_arm_demo)

## X86 CPU

Paddle Lite 当前支持 `AVX` 及 `FMA` 指令集的 X86 CPU，正在完善 `SSE` 指令集的实现。因为 X86 设备众多，不一一列举。
- 下面列举判断当前 CPU 是否同时支持 AVX 及 FMA 指令集（判断 Paddle Lite 是否支持该 CPU）的方法：
  - 目标设备是 linux：执行命令 `cat /proc/cpuinfo` 查看
  - 目标设备是 windows：利用免费工具 `CPU-Z` 查看

### X86 CPU 环境准备及示例程序

- [Paddle Lite 使用 X86 预测部署](../demo_guides/x86)

## 移动端 GPU

Paddle Lite 支持多种移动端 GPU，包括 ARM Mali、Qualcomm Adreno、Apple A Series 等系列 GPU 设备，支持列表如下：
- ARM Mali G 系列
- Qualcomm Adreno 系列
- Apple A 系列

### 移动端 GPU 环境准备及示例程序

- [Paddle Lite 使用 OpenCL 预测部署](../demo_guides/opencl)

## FPGA

Paddle Lite 支持 **百度 FPGA**，支持列表如下：
- 百度 Edgeboard 系列：ZU9, ZU5, ZU3

Paddle Lite 支持 **英特尔 (Intel) FPGA**，支持列表如下：
- 支持芯片：英特尔 FPGA Cyclone V 系列芯片
- 支持设备：
  - 海运捷讯 C5MB（英特尔 FPGA Cyclone V）开发板
  - 海运捷讯 C5CB（英特尔 FPGA Cyclone V）开发板
  - 海运捷讯 C5TB（英特尔 FPGA Cyclone V）开发板

### FPGA 环境准备及示例程序

- 百度 FPGA: [Paddle Lite 使用 FPGA 预测部署](../demo_guides/fpga)
- 英特尔 (Intel) FPGA: [Paddle Lite 使用英特尔 FPGA 预测部署](../demo_guides/intel_fpga)


## 昆仑芯 (kunlunxin) XPU

Paddle Lite 支持昆仑芯 XPU，支持列表如下：
- 昆仑芯 818-100 芯片
- 昆仑芯 818-300 芯片

### 昆仑芯 (kunlunxin) XPU 环境准备及示例程序

- [Paddle Lite 使用昆仑芯 XPU 预测部署](../demo_guides/kunlunxin_xpu)

## 华为 (Huawei) 麒麟 NPU

Paddle Lite 支持华为达芬奇架构麒麟 NPU，支持列表如下：
- 支持芯片：Kirin 810/990/985/9000
- 支持设备：
  * Kirin 9000：HUAWEI Mate40 pro 系列
  * Kirin 9000E：HUAWEI Mate40 系列
  * Kirin 990 5G：HUAWEI Mate30 pro 系列，P40 pro 系列
  * Kirin 990：HUAWEI Mate30 系列, 荣耀 V20 系列, nova6 系列，P40 系列，Mate Xs
  * Kirin 985：HUAWEI nova7 5G，nova7 Pro 5G，荣耀 30
  * Kirin 820：HUAWEI nova7 SE 5G，荣耀 30S
  * Kirin 810：HUAWEI nova5 系列，nova6 SE，荣耀 9X 系列，荣耀 Play4T Pro

### 华为 (Huawei) 麒麟 NPU 环境准备及示例程序

- [Paddle Lite 使用华为麒麟 NPU 预测部署](../demo_guides/huawei_kirin_npu)

## 华为 (Huawei) 昇腾 NPU

Paddle Lite 已支持华为昇腾 NPU（Ascend310、Ascend710和Ascend910）在 X86 和 ARM 服务器上进行预测部署
- 支持芯片
  * Ascend 310 （CANN Version ≥  3.3.0）
  * Ascend 710 （CANN Version ≥  5.0.2.alpha005)
  * Ascend 910 （CANN Version ≥  5.0.2.alpha005)
- 已验证的支持设备
  * Atlas 300I 推理卡（型号：3000/3010)（CANN Version ≥  3.3.0）
  * Atlas 200 DK 开发者套件（CANN Version ≥  3.3.0）
  * Atlas 800 推理服务器（型号：3000/3010）（CANN Version ≥  3.3.0）
  * Atlas 300I Pro（CANN Version ≥ 5.0.2.alpha005)
  * Atlas 300T 训练卡（CANN Version ≥  5.0.2.alpha005)

### 华为 (Huawei) 昇腾 NPU 环境准备及示例程序

- [Paddle Lite 使用华为昇腾 NPU 预测部署](../demo_guides/huawei_ascend_npu)

## 瑞芯微 (Rockchip) NPU

Paddle Lite 支持 瑞芯微 (Rockchip) NPU，支持列表如下：
- 支持芯片：RK1808, RK1806，暂不支持 RK3399Pro
- 支持设备：RK1808/1806 EVB，TB-RK1808S0

### 瑞芯微 (Rockchip) NPU 环境准备及示例程序

- [Paddle Lite 使用瑞芯微 NPU 预测部署](../demo_guides/rockchip_npu)

## 英特尔 (Intel) OpenVINO

Paddle Lite 支持英特尔 OpenVINO 预测部署
- 支持设备:
  - CPU, CPU 型号可查看 OpenVINO [官方数据](https://github.com/openvinotoolkit/openvino#supported-hardware-matrix)

### 英特尔 (Intel) OpenVINO 示例程序

- [Paddle Lite 使用英特尔 OpenVINO 预测部署](../demo_guides/intel_openvino)

## Android NNAPI

Paddle Lite 支持 Android NNAPI，支持列表如下：
- 支持设备：Android 8.1(Oreo) 及以上的终端设备（Android SDK version 需在 27 及以上）

### Android NNAPI 环境准备及示例程序

- [Paddle Lite 使用 Android NNAPI 预测部署](../demo_guides/android_nnapi)

## 联发科 (MediaTek) APU

Paddle Lite 支持 联发科 (MediaTek) APU，支持列表如下：
- 支持芯片：MT8168/MT8175，及其他智能芯片
- 支持设备：MT8168-P2V1 Tablej

### 联发科 (MediaTek) APU 环境准备及示例程序

- [Paddle Lite 使用联发科 APU 预测部署](../demo_guides/mediatek_apu)

## 芯原 TIM-VX

Paddle Lite 支持 芯原 TIM-VX，支持列表如下：
- 支持芯片：搭载了芯原 NPU 的 SoC，驱动版本需为 6.4.4.3

### 芯原 TIM-VX 环境准备及示例程序

- [Paddle Lite 使用 芯原 TIM-VX 预测部署](../demo_guides/verisilicon_timvx)

## 晶晨（Amlogic）NPU

Paddle Lite 支持 晶晨（Amlogic）NPU, 支持列表如下：
- 支持芯片：C308X，A311D，S905D3(Android 版本)

### 晶晨（Amlogic）NPU 环境准备及示例程序

- [Paddle Lite 使用 Amlogic NPU 预测部署](../demo_guides/amlogic_npu)

## 颖脉 (Imagination) NNA

Paddle Lite 支持 颖脉 (Imagination) NNA，支持列表如下：
- 支持芯片：紫光展锐虎贲 T7510
- 支持设备：海信 F50，Roc1 开发板(基于 T7510 的微型电脑主板)

### 颖脉 (Imagination) NNA 环境准备及示例程序

- [Paddle Lite 使用颖脉 NNA 预测部署](../demo_guides/imagination_nna)

## 比特大陆（Bitmain）TPU

Paddle Lite 支持比特大陆（Bitmain）TPU，支持列表如下：
- 支持芯片：Sophon BM1682，Sophon BM1684
- 支持设备：
  * Sophon SC3 加速卡 (BM1682 X86 PCI-E)
  * Sophon SC5 加速卡 (BM1684 X86 PCI-E)

### 比特大陆（Bitmain）TPU 环境准备及示例程序

- [Paddle Lite 使用 Bitmain Sophon BM1682/BM1684 预测部署](../demo_guides/bitmain)
