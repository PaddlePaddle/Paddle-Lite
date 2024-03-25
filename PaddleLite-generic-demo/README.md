# PaddleLite-generic-demo
Classic demo provided for AI accelerators adapted in Paddle Lite. Please check the Paddle Lite's document [https://www.paddlepaddle.org.cn/lite](https://www.paddlepaddle.org.cn/lite) for more details.

## Model test
```
cd model_test/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 cpu localhost 9031 root root
  ```
### OpenCL
-  Arm CPU + Mali/Adreno GPU (Android)
  ```
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a opencl UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android armeabi-v7a opencl UQG0220A15000356
  ```
- Arm CPU + Mali GPU (Linux)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 opencl 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux armhf opencl 192.168.100.13 22 root rockchip
  ```
- x86 CPU + Intel/Nvidia GPU (Linux)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 opencl localhost 9031 root root
  ```
### Huawei Kirin NPU
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a huawei_kirin_npu UQG0220A15000356
  ```
### MediaTek APU
- MediaTek MT8618 Tablet (Android)
  ```
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ./run_with_adb.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ```
### Imagination NNA
- ROC1 (Linux)
  ```
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ```
### Huawei Ascend NPU
- x86 CPU + Huawei Atlas 300C(3010) (Ubuntu)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 huawei_ascend_npu localhost 9031 root root
  ```
- Arm CPU + Huawei Atlas 300C(3000) (CentOS)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 huawei_ascend_npu localhost 9031 root root
  ```
### Verisilicon TIM-VX (Rockchip NPU and Amlogic NPU)
- Khadas VIM3L (Android)
  ```
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android armeabi-v7a verisilicon_timvx c8631471d5cd
  ```
- Khadas VIM3 (Linux)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ```
- RK1808EVB (Linux)
  ```
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux arm64 verisilicon_timvx a133d8abb26137b2
  ```
- Toybirck TB-RK1808S0 (Linux)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ```
- RV1109 (Linux)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 xpu localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 xpu localhost 9031 root root
  ```
### Kunlunxin XPU with XTCL
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 kunlunxin_xtcl localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 kunlunxin_xtcl localhost 9031 root root
  ```
### Cambricon MLU
- x86 CPU + Cambricon MLU 370 (Ubuntu)
  ```
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 cambricon_mlu localhost 9031 root root
  ```
### Android NNAPI
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 android armeabi-v7a android_nnapi UQG0220A15000356
  ```
### Qualcomm QNN
- x86 CPU (QNN Simulator, Ubuntu)
  ```
  unset FILE_TRANSFER_COMMAND

  ./run.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 qualcomm_qnn
  ./run.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run.sh conv_add_144_192_int8_per_layer 1,3,144,192 float32 float32 linux amd64 qualcomm_qnn
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,144,192 float32 float32 linux amd64 qualcomm_qnn localhost 9031 root root
  ```
- Qualcomm 8295P EVK (Android)
  ```
  adb -s 858e5789 root

  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,144,192 float32 float32 android arm64-v8a qualcomm_qnn 858e5789
  ```
- Qualcomm 8295P EVK (QNX)
  ```
  export FILE_TRANSFER_COMMAND=lftp
  adb -s 858e5789 root

  rm -rf ../assets/models/cache
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,144,192 float32 float32 android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,144,192 float32 float32 qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache
  ```


## Image classification demo based on MobileNet, ResNet etc.
```
cd image_classification_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 cpu localhost 9031 root root
  ```
### OpenCL
-  Arm CPU + Mali/Adreno GPU (Android)
  ```
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a opencl UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a opencl UQG0220A15000356
  ```
- Arm CPU + Mali GPU (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 opencl 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux armhf opencl 192.168.100.13 22 root rockchip
  ```
- x86 CPU + Intel/Nvidia GPU (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 opencl localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 opencl localhost 9031 root root
  ```
### Huawei Kirin NPU
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a huawei_kirin_npu UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a huawei_kirin_npu UQG0220A15000356
  ```
### MediaTek APU
- MediaTek MT8168 Tablet (Android)
  ```
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ./run_with_adb.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ```
### Imagination NNA
- ROC1 (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ```
### Huawei Ascend NPU
- Intel CPU + Huawei Atlas 300C(3010) (Lenovo P720 + Ubuntu 16.04)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 huawei_ascend_npu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 huawei_ascend_npu localhost 9031 root root
  ```
- Kunpeng 920 + Huawei Atlas 300C(3000) (Kunpeng 920 Desktop PC + CentOS 7.6)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 huawei_ascend_npu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 huawei_ascend_npu localhost 9031 root root
  ```
### Verisilicon TIM-VX (Rockchip NPU and Amlogic NPU)
- Khadas VIM3L (Android)
  ```
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ```
- Khadas VIM3 (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ```
- RK1808EVB (Linux)
  ```
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ```
- Toybirck TB-RK1808S0 (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ```
- RV1109 (Linux)
  ```
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 xpu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 xpu localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 xpu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 xpu localhost 9031 root root
  ```
### Kunlunxin XPU with XTCL
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 kunlunxin_xtcl localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 kunlunxin_xtcl localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 kunlunxin_xtcl localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 kunlunxin_xtcl localhost 9031 root root
  ```
### Cambricon MLU
- x86 CPU + Cambricon MLU 370 (Lenovo P720 + Ubuntu 16.04)
  ```
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 cambricon_mlu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 cambricon_mlu localhost 9031 root root
  ```
### Android NNAPI
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ```
### Qualcomm QNN
- x86 CPU (QNN Simulator, Ubuntu)
  ```
  unset FILE_TRANSFER_COMMAND

  ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn
  ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux amd64 qualcomm_qnn
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn
  ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run.sh resnet50_int8_224_per_layer imagenet_224.txt test linux amd64 qualcomm_qnn
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ```
- Qualcomm 8295P EVK (Android)
  ```
  adb -s 858e5789 root

  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789
  ```
- Qualcomm 8295P EVK (QNX)
  ```
  export FILE_TRANSFER_COMMAND=lftp
  adb -s 858e5789 root

  rm -rf ../assets/models/cache
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache
  ```

## Object detection demo based on SSD, YOLO etc.
```
cd object_detection_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 cpu localhost 9031 root root
  ```
### OpenCL
-  Arm CPU + Mali/Adreno GPU (Android)
  ```
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a opencl UQG0220A15000356
  ```
- Arm CPU + Mali GPU (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 opencl 192.168.100.30 22 khadas khadas
  ```
- x86 CPU + Intel/Nvidia GPU (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 opencl localhost 9031 root root
  ```
### Huawei Kirin NPU
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a huawei_kirin_npu UQG0220A15000356
  ```
### MediaTek APU
- MediaTek MT8618 Tablet (Android)
  ```
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ```
### Imagination NNA
- ROC1 (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ```
### Huawei Ascend NPU
- x86 CPU + Huawei Atlas 300C(3010) (Ubuntu)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 huawei_ascend_npu localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 huawei_ascend_npu localhost 9031 root root
  ```
- Arm CPU + Huawei Atlas 300C(3000) (CentOS)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 huawei_ascend_npu localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux arm64 huawei_ascend_npu localhost 9031 root root
  ```
### Verisilicon TIM-VX (Rockchip NPU and Amlogic NPU)
- Khadas VIM3L (Android)
  ```
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ```
- Khadas VIM3 (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ```
- RK1808EVB (Linux)
  ```
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt linux arm64 verisilicon_timvx a133d8abb26137b2
  ```
- Toybirck TB-RK1808S0 (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ```
- RV1109 (Linux)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 xpu localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 xpu localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 xpu localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux arm64 xpu localhost 9031 root root
  ```
### Kunlunxin XPU with XTCL
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 kunlunxin_xtcl localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 kunlunxin_xtcl localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 kunlunxin_xtcl localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux arm64 kunlunxin_xtcl localhost 9031 root root
  ```
### Cambricon MLU
- x86 CPU + Cambricon MLU 370 (Ubuntu)
  ```
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 cambricon_mlu localhost 9031 root root
  ```
### Android NNAPI
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ```
### Qualcomm QNN
- x86 CPU (QNN Simulator, Ubuntu)
  ```
  unset FILE_TRANSFER_COMMAND

  ./run.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 qualcomm_qnn
  ./run.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux amd64 qualcomm_qnn
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 qualcomm_qnn
  ./run.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ```
- Qualcomm 8295P EVK (Android)
  ```
  adb -s 858e5789 root

  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ```
- Qualcomm 8295P EVK (QNX)
  ```
  export FILE_TRANSFER_COMMAND=lftp
  adb -s 858e5789 root

  rm -rf ../assets/models/cache
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ```

## Keypoint detection demo based on PP-TinyPose etc.
```
cd keypoint_detection_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 cpu localhost 9031 root root
  ```
### OpenCL
-  Arm CPU + Mali/Adreno GPU (Android)
  ```
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a opencl UQG0220A15000356
  ```
- Arm CPU + Mali GPU (Linux)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux arm64 opencl 192.168.100.30 22 khadas khadas
  ```
- x86 CPU + Intel/Nvidia GPU (Linux)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 opencl localhost 9031 root root
  ```
### Huawei Kirin NPU
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a huawei_kirin_npu UQG0220A15000356
  ```
### MediaTek APU
- MediaTek MT8618 Tablet (Android)
  ```
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ```
### Imagination NNA
- ROC1 (Linux)
  ```
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ```
### Huawei Ascend NPU
- x86 CPU + Huawei Atlas 300C(3010) (Ubuntu)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 huawei_ascend_npu localhost 9031 root root
  ```
- Arm CPU + Huawei Atlas 300C(3000) (CentOS)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux arm64 huawei_ascend_npu localhost 9031 root root
  ```
### Verisilicon TIM-VX (Rockchip NPU and Amlogic NPU)
- Khadas VIM3L (Android)
  ```
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ```
- Khadas VIM3 (Linux)
  ```
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ```
- RK1808EVB (Linux)
  ```
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ```
- Toybirck TB-RK1808S0 (Linux)
  ```
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ```
- RV1109 (Linux)
  ```
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 xpu localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux arm64 xpu localhost 9031 root root
  ```
### Kunlunxin XPU with XTCL
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 kunlunxin_xtcl localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux arm64 kunlunxin_xtcl localhost 9031 root root
  ```
### Cambricon MLU
- x86 CPU + Cambricon MLU 370 (Ubuntu)
  ```
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 cambricon_mlu localhost 9031 root root
  ```
### Android NNAPI
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ```
### Qualcomm QNN
- x86 CPU (QNN Simulator, Ubuntu)
  ```
  unset FILE_TRANSFER_COMMAND

  ./run.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 qualcomm_qnn
  ./run.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ```
- Qualcomm 8295P EVK (Android)
  ```
  adb -s 858e5789 root

  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ```
- Qualcomm 8295P EVK (QNX)
  ```
  export FILE_TRANSFER_COMMAND=lftp
  adb -s 858e5789 root

  rm -rf ../assets/models/cache
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ```

## Semantic segmentation demo based on PP-LiteSeg/PP-HumanSeg etc.
```
cd semantic_segmentation_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android arm64-v8a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ```
- x86 CPU (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux amd64 cpu localhost 9031 root root
  ```
### OpenCL
-  Arm CPU + Mali/Adreno GPU (Android)
  ```
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a opencl UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a opencl UQG0220A15000356
  ```
- Arm CPU + Mali GPU (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 opencl 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 opencl 192.168.100.30 22 khadas khadas
  ```
- x86 CPU + Intel/Nvidia GPU (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 opencl localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 opencl localhost 9031 root root
  ```
### Huawei Kirin NPU
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a huawei_kirin_npu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a huawei_kirin_npu UQG0220A15000356
  ```
### MediaTek APU
- MediaTek MT8618 Tablet (Android)
  ```
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android armeabi-v7a mediatek_apu 0123456789ABCDEF
  ```
### Imagination NNA
- ROC1 (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ```
### Huawei Ascend NPU
- x86 CPU + Huawei Atlas 300C(3010) (Ubuntu)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 huawei_ascend_npu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 huawei_ascend_npu localhost 9031 root root
  ```
- Arm CPU + Huawei Atlas 300C(3000) (CentOS)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 huawei_ascend_npu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 huawei_ascend_npu localhost 9031 root root
  ```
### Verisilicon TIM-VX (Rockchip NPU and Amlogic NPU)
- Khadas VIM3L (Android)
  ```
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android armeabi-v7a verisilicon_timvx c8631471d5cd
  ```
- Khadas VIM3 (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
  ```
- RK1808EVB (Linux)
  ```
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux arm64 verisilicon_timvx a133d8abb26137b2
  ```
- Toybirck TB-RK1808S0 (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux arm64 verisilicon_timvx 192.168.180.8 22 toybrick toybrick
  ```
- RV1109 (Linux)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux armhf verisilicon_timvx 192.168.100.13 22 root rockchip
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 xpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 xpu localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 xpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 xpu localhost 9031 root root
  ```
### Kunlunxin XPU with XTCL
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 kunlunxin_xtcl localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 kunlunxin_xtcl localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 kunlunxin_xtcl localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 kunlunxin_xtcl localhost 9031 root root
  ```
### Cambricon MLU
- x86 CPU + Cambricon MLU 370 (Ubuntu)
  ```
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 cambricon_mlu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 cambricon_mlu localhost 9031 root root
  ```
### Android NNAPI
- Huawei P40pro 5G (Android)
  ```
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android arm64-v8a android_nnapi UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android armeabi-v7a android_nnapi UQG0220A15000356
  ```
### Qualcomm QNN
- x86 CPU (QNN Simulator, Ubuntu)
  ```
  unset FILE_TRANSFER_COMMAND

  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 qualcomm_qnn
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux amd64 qualcomm_qnn
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux amd64 qualcomm_qnn
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 qualcomm_qnn
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 qualcomm_qnn "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux amd64 qualcomm_qnn
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux amd64 qualcomm_qnn

  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 qualcomm_qnn localhost 9031 root root "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux amd64 qualcomm_qnn localhost 9031 root root
  ```
- Qualcomm 8295P EVK (Android)
  ```
  adb -s 858e5789 root

  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789
  ```
- Qualcomm 8295P EVK (QNX)
  ```
  export FILE_TRANSFER_COMMAND=lftp
  adb -s 858e5789 root

  rm -rf ../assets/models/cache
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache

  rm -rf ../assets/models/cache
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android arm64-v8a qualcomm_qnn 858e5789 null cache
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache
  ```
