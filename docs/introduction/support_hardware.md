
# 支持硬件


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

## 移动端GPU
Paddle Lite支持移动端GPU和Nvidia端上GPU设备，支持列表如下：
- ARM Mali G 系列
- Qualcomm Adreno 系列
  
  Nvida tegra系列: tx1, tx2, nano, xavier

## FPGA
Paddle Lite支持FPGA，支持列表如下：
- 百度Edgeboard系列：ZU9, ZU5, ZU3

## 百度 (Baidu) XPU
Paddle Lite支持百度XPU，支持列表如下：
- 百度昆仑818-100芯片
- 百度昆仑818-300芯片

## 华为 (Huawei) NPU
Paddle Lite支持华为达芬奇架构NPU，支持列表如下：
- 支持芯片：Kirin 810/990/985/9000, Ascend 310
- 支持设备：
  * Kirin 990：HUAWEI Mate 30系列, 荣耀 V20系列, nova 6系列，P40系列，Mate Xs
  * Kirin 985：HUAWEI nova 7 5G，nova 7 Pro 5G，荣耀 30
  * Kirin 820：HUAWEI nova 7 SE 5G，荣耀 30S
  * Kirin 810：HUAWEI nova 5系列，nova 6 SE，荣耀 9X系列，荣耀 Play4T Pro
  * Ascend 310：Atlas300推理卡

## 瑞芯微 (Rockchip) NPU
Paddle Lite支持 瑞芯微 (Rockchip) NPU，支持列表如下：
- 支持芯片：RK1808, RK1806，暂不支持RK3399Pro
- 支持设备：RK1808/1806 EVB，TB-RK1808S0

## 联发科 (MediaTek) APU
Paddle Lite支持 联发科 (MediaTek) APU，支持列表如下：
- 支持芯片：MT8168/MT8175，及其他智能芯片
- 支持设备：MT8168-P2V1 Tablet

## 颖脉 (Imagination) NNA
Paddle Lite支持 颖脉 (Imagination) NNA，支持列表如下：
- 支持芯片：紫光展锐虎贲T7510
- 支持设备：海信F50，Roc1开发板（基于T7510的微信电脑主板）
