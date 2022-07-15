# 华为昇腾 NPU 部署示例

Paddle Lite 已支持华为昇腾 NPU（Ascend310、Ascend710和Ascend910）在 X86 和 ARM 服务器上进行预测部署。 目前支持子图接入方式，其接入原理是在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 Ascend NPU 组网 API 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- Ascend 310  （CANN Version ≥  3.3.0）
- Ascend 310P （CANN Version ≥  5.0.2.alpha005)
- Ascend 910  （CANN Version ≥  5.0.2.alpha005)

### 已支持的设备

- Atlas 300I 推理卡（型号：3000/3010)
- Atlas 200 DK 开发者套件
- Atlas 800 推理服务器（型号：3000/3010）
- Atlas 300I Pro（CANN Version ≥ 5.0.2.alpha005)
- Atlas 300T 训练卡（CANN Version ≥  5.0.2.alpha005)

### 已验证支持的版本

- CANN 版本 ≥ 3.3.0
- 固件与驱动版本 ≥ 1.0.9.alpha
- 设备的版本配套关系见 [Ascend 官网](https://www.hiascend.com/hardware/firmware-drivers?tag=community)

### 已验证支持的 Paddle 模型

#### 模型

- 图像分类
  - [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz)
  - [DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz)
  - [DeiT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DeiT/DeiT_base_patch16_224.tar.gz)
  - [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz)
  - [DPN68](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DPN/DPN68.tar.gz)
  - [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz)
  - [GhostNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/GhostNet/GhostNet_x1_0.tar.gz)
  - [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz)
  - [HRNet-W18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/HRNet/HRNet_W18_C.tar.gz)
  - [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz)
  - [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz)
  - [MobileNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz)
  - [MobileNet-v2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz)
  - [MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_large_x1_0.tgz)
  - [MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_small_x1_0.tgz)
  - [PP-LCNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/PPLCNet/PPLCNet_x0_25.tar.gz)
  - [Res2Net50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/Res2Net/Res2Net50_26w_4s.tar.gz)
  - [ResNet-101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz)
  - [ResNet-18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz)
  - [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz)
  - [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz)
  - [SE_ResNet50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SENet/SE_ResNet50_vd.tar.gz)
  - [ShuffleNetV2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ShuffleNetV2_x1_0.tgz)
  - [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz)
  - [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz)
  - [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz)
  - [ViT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/VisionTransformer/ViT_base_patch16_224.tar.gz)
- 目标检测
  - [Picodet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/picodet_m_416_coco.tar.gz)
  - [PP-YOLO_mbv3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_mbv3_large_coco.tar.gz)
  - [PP-YOLO_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_r50vd_dcn_1x_coco.tar.gz)
  - [PPYOLO_tiny](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_tiny_650e_coco.tar.gz)
  - [PP-YOLOv2_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolov2_r50vd_dcn_365e_coco.tar.gz)
  - [SSD-MobileNetV1(1.8)](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)
  - [SSD-MobileNetV1(2.0+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_mobilenet_v1_300_120e_voc.tgz)
  - [SSDLite-MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_large.tar.gz)
  - [SSDLite-MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_small.tar.gz)
  - [SSD-VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_vgg16_300_240e_voc.tgz)
  - [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz)
  - [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz)
  - [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz)
  - [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz)
  - [YOLOv4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz)
- 姿态检测
  - [PP-TinyPose](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/tinypose_128x96.tar.gz)
- 人脸检测
  - [BlazeFace](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/blazeface_1000e.tgz)
  - [FaceBoxes](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/faceboxes.tgz)
- 关键点检测
  - [HigherHRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz)
  - [HRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz)
- 文本检测 & 文本识别 & 端到端检测识别
  - [ch_ppocr_mobile_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_det_infer.tgz)
  - [ch_ppocr_mobile_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_rec_infer.tgz)
  - [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz)
  - [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz)
  - [ch_PP-OCRv2_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_det_infer.tar.gz)
  - [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz)
  - [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz)
  - [e2e_server_pgnetA](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/e2e_server_pgnetA.tar.gz)
- 自然语言处理 & 语义理解
  - [BERT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/bert_base_uncased.tgz)
  - [ERNIE](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_1.0.tgz)
  - [ERNIE-TINY](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_tiny.tgz)
  - [Transformer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/transformer.tar.gz)
- 生成网络
  - [ESRGAN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tgz)
- 推荐系统
  - [DeepFM](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/deepfm.tar.gz)
  - [NAML](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/naml.tar.gz)
  - [NCF](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/ncf.tar.gz)
  - [Wide&Deep](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/wide_deep.tar.gz)
- 图像分割
  - [BiseNetV2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/bisenet.tar.gz)
  - [DeepLabV3+(CityScapes)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.tar.gz)
  - [PP-HumanSeg-Lite](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/pphumanseg_lite_generic_192x192_with_softmax.tar.gz)
  - [PP-HumanSeg-Server(DeepLabV3+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.tar.gz)
  - [SegFormer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/segformer.tar.gz)
  - [STDCSeg](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/stdcseg.tar.gz)
  - [U-Net](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/unet_cityscapes_1024x512_160k.tgz)
- 视频分类
  - [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz)


- [开源模型支持列表](../quick_start/support_model_list)



#### 性能（基于 CANN 3.3.0 版本测试）

性能仅供参考,以实际运行效果为准。


| 模型                                                         | Intel CPU性能 (ms) | x86 + Ascend310 性能 (ms） | 鲲鹏 920 CPU 性能 (ms) | 鲲鹏920 + Ascend310 性能 (ms) |
| ------------------------------------------------------------ | ------------------ | -------------------------- | ---------------------- | ----------------------------- |
| [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz) | 81.16              | 4.32                       | 67.39                  | 3.64                          |
| [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz) | 529.45             | 7.81                       | 157.62                 | 6.23                          |
| [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz) | 167.59             | 21.08                      | 85.98                  | 7.14                          |
| [ERNIE-TINY](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_tiny.tgz) | -                  | 12.31                      | 347.47                 | 10.72                         |
| [ESRGAN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tgz) | 41774.81           | 103.75                     | 9755.14                | 103.06                        |
| [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz) | 104.55             | 2.65                       | 71.24                  | 2.40                          |
| [HigherHRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz) | 6760.83            | 386.02                     | 6482.55                | 396.17                        |
| [HRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz) | 1245.54            | 12.58                      | 752.93                 | 12.08                         |
| [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz) | 415.00             | 5.91                       | 159.40                 | 3.46                          |
| [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz) | 1082.00            | 9.00                       | 706.72                 | 6.72                          |
| [MobileNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz) | 33.88              | 3.06                       | 35.08                  | 1.43                          |
| [MobileNet-v2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz) | 25.67              | 3.82                       | 25.27                  | 1.74                          |
| [MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_large_x1_0.tgz) | 43.53              | 8.96                       | 20.46                  | 2.83                          |
| [MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_small_x1_0.tgz) | 29.38              | 6.38                       | 8.66                   | 2.24                          |
| [ch_ppocr_mobile_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_det_infer.tgz) | 359.16             | 10.75                      | 129.01                 | 10.78                         |
| [ch_ppocr_mobile_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_rec_infer.tgz) | 32.39              | 8.75                       | 15.05                  | 5.87                          |
| [ResNet-101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz) | 528.43             | 7.33                       | 401.14                 | 4.27                          |
| [ResNet-18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz) | 136.10             | 2.84                       | 89.41                  | 1.59                          |
| [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz) | 258.21             | 5.66                       | 229.04                 | 2.87                          |
| [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz) | 353.43             | 5.32                       | 261.28                 | 3.10                          |
| [ShuffleNetV2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ShuffleNetV2_x1_0.tgz) | 37.93              | 5.41                       | 14.12                  | 4.06                          |
| [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz) | 51.12              | 3.06                       | 40.50                  | 1.49                          |
| [SSD-MobileNetV1(1.8)](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz) | 64.82              | 6.56                       | 70.40                  | 6.19                          |
| [SSD-MobileNetV1(2.0+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_mobilenet_v1_300_120e_voc.tgz) | 70.72              | 8.79                       | 73.16                  | 8.17                          |
| [SSD-VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_vgg16_300_240e_voc.tgz) | 1999.32            | 27.35                      | 1001.06                | 25.93                         |
| [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz) | 1204.23            | 10.53                      | 443.20                 | 9.43                          |
| [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz) | 1512.82            | 11.50                      | 667.20                 | 10.23                         |
| [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz) | 4241.35            | 38.46                      | 2806.42                | 34.47                         |
| [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz) | 1181.56            | 27.20                      | 941.62                 | 23.13                         |
| [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz) | 1396.57            | 27.97                      | 789.02                 | 27.06                         |
| [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz) | 15521.84           | 55.41                      | 2444.08                | 48.96                         |
| [YOLOv4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz) | 7389.54            | 77.80                      | 5503.77                | 71.78                         |



#### 性能（基于 CANN 5.1.RC1.alpha001 版本测试）

性能仅供参考,以实际运行效果为准。

| 模型                                                         | Intel CPU性能 (ms) | x86 + Ascend310 性能 (ms） | 鲲鹏 920 CPU 性能 (ms) | 鲲鹏920 + Ascend310 性能 (ms) |
| ------------------------------------------------------------ | ------------------ | -------------------------- | ---------------------- | ----------------------------- |
| [DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz) | 11791.1            | 4.2608                     | 347.008                | 3.3928                        |
| [DeiT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DeiT/DeiT_base_patch16_224.tar.gz) | -                  | 20.4874                    | 1593.23                | 19.9074                       |
| [DPN68](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DPN/DPN68.tar.gz) | 199.321            | 4.7786                     | 171.64                 | 4.1362                        |
| [GhostNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/GhostNet/GhostNet_x1_0.tar.gz) | 26.6712            | 5.8512                     | 16.5946                | 3.7472                        |
| [HRNet-W18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/HRNet/HRNet_W18_C.tar.gz) | 271.791            | 8.3446                     | 245.833                | 7.2004                        |
| [PP-LCNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/PPLCNet/PPLCNet_x0_25.tar.gz) | 3.5006             | 2.1592                     | 2.893                  | 1.5552                        |
| [Res2Net50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/Res2Net/Res2Net50_26w_4s.tar.gz) | 652.982            | 8.1164                     | 246.673                | 7.0866                        |
| [SE_ResNet50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SENet/SE_ResNet50_vd.tar.gz) | 231.064            | 12.3792                    | 236.161                | 5.3052                        |
| [ViT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/VisionTransformer/ViT_base_patch16_224.tar.gz) | -                  | 20.4644                    | 1556.62                | 19.9086                       |
| [Picodet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/picodet_m_416_coco.tar.gz) | 94.617             | 14.938                     | 89.9                   | 12.083                        |
| [PP-YOLO_mbv3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_mbv3_large_coco.tar.gz) | 56.4607            | 8.2009                     | 92.8885                | 4.5504                        |
| [PP-YOLO_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_r50vd_dcn_1x_coco.tar.gz) | 6487.56            | 57.5038                    | 3213.65                | 51.7323                       |
| [PP-YOLO_tiny](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_tiny_650e_coco.tar.gz) | 30.1026            | 11.1346                    | 48.245                 | 6.33                          |
| [PP-YOLOv2_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolov2_r50vd_dcn_365e_coco.tar.gz) | 10019.3            | 83.2727                    | 5727.97                | 124.624                       |
| [SSDLite-MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_large.tar.gz) | 63.9276            | 8.205                      | 55.6868                | 6.6285                        |
| [SSDLite-MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_small.tar.gz) | 30.1444            | 12.0989                    | 22.8334                | 7.7404                        |
| [PP-TinyPose](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/tinypose_128x96.tar.gz) | 25.685             | 8.898                      | 12.009                 | 9.394                         |
| [BlazeFace](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/blazeface_1000e.tgz) | 217.842            | 51.311                     | 162.828                | 29.823                        |
| [FaceBoxes](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/faceboxes.tgz) | 503.529            | 82.158                     | 285.394                | 51.057                        |
| [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz) | 1046.45            | 21.556                     | 1323.83                | 18.661                        |
| [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz) | 283.479            | 55.651                     | 730.876                | 41.489                        |
| [ch_PP-OCRv2_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_det_infer.tar.gz) | 167.616            | 15.24                      | 202.873                | 13.491                        |
| [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz) | 47.577             | 14.533                     | 48.63                  | 9.43                          |
| [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz) | 12.146             | 9.921                      | 10.125                 | 4.751                         |
| [e2e_server_pgnetA](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/e2e_server_pgnetA.tar.gz) | 8432               | 84.441002                  | 10715                  | 107.522003                    |
| [BERT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/bert_base_uncased.tgz) | -                  | 13.545                     | 233.982                | 13.489                        |
| [ERNIE](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_1.0.tgz) | -                  | 13.249                     | 211.621                | 12.763                        |
| [ERNIE-TINY](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_tiny.tgz) | -                  | 6.02                       | 100.079                | 5.78                          |
| [Transformer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/transformer.tar.gz) | -                  | -                          | 899.997                | 665.344                       |
| [DeepFM](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/deepfm.tar.gz) | 0.397              | 1.426                      | 0.363                  | 1.394                         |
| [NAML](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/naml.tar.gz) | -                  | 2.605                      | 6.201                  | 2.211                         |
| [NCF](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/ncf.tar.gz) | 0.011              | 0.727                      | 0.031                  | 0.464                         |
| [Wide&Deep](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/wide_deep.tar.gz) | 0.399              | 1.448                      | 0.323                  | 1.339                         |
| [BiseNetV2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/bisenet.tar.gz) | 3670.16            | 133.841                    | 3880.19                | 124.746                       |
| [DeepLabV3+(CityScapes)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.tar.gz) | 52215.8            | 671.86                     | 76647.2                | 653.046                       |
| [PP-HumanSeg-Lite](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/pphumanseg_lite_generic_192x192_with_soft.tar.gz) | 44.053             | 12.602                     | 15.502                 | 12.08                         |
| [PP-HumanSeg-Server(DeepLabV3+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_soft.tar.gz) | 6569.85            | 125.655                    | 7923.12                | 123.139                       |
| [SegFormer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/segformer.tar.gz) | -                  | 82.789                     | 1440.23                | 83.664                        |
| [STDCSeg](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/stdcseg.tar.gz) | 457.75             | 21.626                     | 529.451                | 17.376                        |
| [U-Net](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/unet_cityscapes_1024x512_160k.tgz) | 50527.6            | 986.409                    | 34502                  | 977.992                       |
| [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz) | 69786.5            | 807.352                    | 53438                  | 644.648                       |
| [resnet50_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/resnet50_int8_224_per_layer.tar.gz) | -                  | -                          | -                      | 4.58                          |
| [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz) | -                  | -                          | -                      | 2.53                          |

### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示（以社区版 CANN 5.1.RC1.alpha001 为例）

测试设备（Atlas300I 推理卡）

<img src="https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/atlas300I.jpg" alt="Huawei_Ascend_NPU" style="zoom: 33%;" />

### 准备设备环境（如 ubuntu18.04-x86_64）

- Atlas 300I 推理卡[规格说明书](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300-ai)
- 安装Atlas 300I 推理卡的驱动和固件包（Driver 和 Firmware)
- 配套驱动和固件包下载：
  - https://www.hiascend.com/hardware/firmware-drivers?tag=community（社区版）
  - https://www.hiascend.com/hardware/firmware-drivers?tag=commercial（商业版）
  - 驱动：A300-3010-npu-driver_21.0.4_linux-x86_64.run
  - 固件：A300-3010-npu-firmware_1.80.22.2.220.run
- 安装驱动和固件包：

```shell
# 增加可执行权限
$ chmod +x *.run
# 安装驱动和固件包
$ ./A300-3010-npu-driver_21.0.4_linux-x86_64.run --full
$ ./A300-3010-npu-firmware_1.80.22.2.220.run --full
# 重启服务器
$ reboot
# 查看驱动信息，确认安装成功
$ npu-smi info
```

- 更多系统和详细信息见[昇腾硬件产品文档](https://www.hiascend.com/document?tag=hardware)

### 准备本地编译环境

- 为了保证编译环境一致，建议使用Docker开发环境进行配置；

- 若想使用新版本的 CANN，请自行更新 Dockerfile 文件内的 CANN 下载路径，当前 Dockerfile 内默认为 [CANN 5.1.RC1.alpha001](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/5.1.RC1.alpha001/Ascend-cann-toolkit_5.1.RC1.alpha001_linux-x86_64.run)；

- for arm64

  ```shell
  # 下载 Dockerfile
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/kunpeng920_arm/Ascend_ubuntu18.04_aarch64_5.1.rc1.alpha001.Dockerfile
  # 通过 Dockerfile 生成镜像
  $ docker build --network=host -f Ascend_ubuntu18.04_aarch64_5.1.rc1.alpha001.Dockerfile -t paddlelite/ascend_aarch64:cann_5.1.1.alpha001 .
  # 创建容器
  $ docker run -itd --privileged --name=ascend-aarch64 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_aarch64:cann_5.1.1.alpha001 /bin/bash
  # 进入容器
  $ docker exec -it ascend-aarch64 /bin/bash
  # 确认容器的 Ascend 环境是否创建成功
  $ npu-smi info
  ```

- for amd64

  ```shell
  # 下载 Dockerfile
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/intel_x86/Ascend_ubuntu18.04_x86_5.1.rc1.alpha001.Dockerfile
  # 通过 Dockerfile 生成镜像
  $ docker build --network=host -f Ascend_ubuntu18.04_x86_5.1.rc1.alpha001.Dockerfile -t paddlelite/ascend_x86:cann_5.1.1.alpha001 .
  # 创建容器
  $ docker run -itd --privileged --name=ascend-x86 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_x86:cann_5.1.1.alpha001 /bin/bash
  # 进入容器
  $ docker exec -it ascend-x86 /bin/bash
  # 确认容器的 Ascend 环境是否创建成功
  $ npu-smi info
  ```

  

### 运行图像分类示例程序

- 下载示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)(该 demo 的默认预编译库基于 CANN 5.1.RC1.alpha001 版本)，解压后清单如下：

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
            - resnet50_fp32_224 # Paddle non-combined 格式的 resnet50 float32 模型
              - __model__ # Paddle fluid 模型组网文件，可拖入 https://lutzroeder.github.io/netron/ 进行可视化显示网络结构
              - bn2a_branch1_mean # Paddle fluid 模型参数文件
              - bn2a_branch1_scale
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.amd64 # 已编译好的，适用于 amd64
            - image_classification_demo # 已编译好的，适用于 amd64 的示例程序
          - build.linux.arm64 # 已编译好的，适用于 arm64
            - image_classification_demo # 已编译好的，适用于 arm64 的示例程序
            ...
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序 ssh 运行脚本
          - run_with_adb.sh # 示例程序 adb 运行脚本
      - libs
        - PaddleLite
          - android
            - arm64-v8a
            - armeabi-v7a
          - linux
            - amd64
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - huawei_ascend_npu # 华为昇腾 NPU CANN 库、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libhuawei_ascend_npu.so # NNAdapter device HAL 库
                - libiomp5.so # Intel OpenMP 库
                - libmklml_intel.so # Intel MKL 库
                - libmklml_gnu.so # GNU MKL 库
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - arm64
              - include
              - lib
            - armhf
              ...
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
  ```

  

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

- 执行以下命令比较 mobilenet_v1_fp32_224 模型的性能和结果；

  ```shell
  运行 mobilenet_v1_fp32_224 模型
    
  For amd64
  (intel x86 cpu only)
  $ ./run.sh mobilenet_v1_fp32_224 linux amd64
      warmup: 1 repeat: 1, average: 44.949001 ms, max: 44.949001 ms, min: 44.949001 ms
      results: 3
      Top0  tabby, tabby cat - 0.529132
      Top1  Egyptian cat - 0.419680
      Top2  tiger cat - 0.045172
      Preprocess time: 1.017000 ms
      Prediction time: 44.949001 ms
      Postprocess time: 0.171000 ms
  (intel x86 cpu + ascend npu)
  $ ./run.sh mobilenet_v1_fp32_224 linux amd64 huawei_ascend_npu
      warmup: 1 repeat: 1, average: 2.079000 ms, max: 2.079000 ms, min: 2.079000 ms
      results: 3
      Top0  tabby, tabby cat - 0.529785
      Top1  Egyptian cat - 0.418945
      Top2  tiger cat - 0.045227
      Preprocess time: 1.132000 ms
      Prediction time: 2.079000 ms
      Postprocess time: 0.251000 ms
  
  For arm64
  (鲲鹏 920 cpu only)
  $ ./run.sh mobilenet_v1_fp32_224 linux arm64
      warmup: 1 repeat: 1, average: 34.160999 ms, max: 34.160999 ms, min: 34.160999 ms
      results: 3
      Top0  tabby, tabby cat - 0.529131
      Top1  Egyptian cat - 0.419681
      Top2  tiger cat - 0.045173
      Preprocess time: 0.571000 ms
      Prediction time: 34.160999 ms
      Postprocess time: 0.081000 ms
  (鲲鹏 920 cpu + ascend npu)
  $ ./run.sh mobilenet_v1_fp32_224 linux arm64 huawei_ascend_npu
      warmup: 1 repeat: 1, average: 1.555000 ms, max: 1.555000 ms, min: 1.555000 ms
      results: 3
      Top0  tabby, tabby cat - 0.529785
      Top1  Egyptian cat - 0.418945
      Top2  tiger cat - 0.045227
      Preprocess time: 0.605000 ms
      Prediction time: 1.555000 ms
      Postprocess time: 0.093000 ms
  ```

- 如果需要更改测试模型为 resnet50，可以将 `run.sh` 里的 MODEL_NAME 改成 resnet50_fp32_224，或执行命令：

  ```shell
  (intel x86 cpu + ascend npu)
  $ ./run.sh resnet50_fp32_224 linux amd64 huawei_ascend_npu
  (鲲鹏 920 cpu + ascend npu)
  $ ./run.sh resnet50_fp32_224 linux arm64 huawei_ascend_npu
  ```

- 如果需要更改测试图片，请将图片拷贝到 **`PaddleLite-generic-demo/image_classification_demo/assets/images`** 目录下，修改并执行 **`convert_to_raw_image.py`** 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可；

- 如果需要重新编译示例程序，直接运行

  ```shell
  # amd64
  $ ./build.sh linux amd64
  # arm64
  $ ./build.sh linux arm64
  ```

### 更新支持华为昇腾 NPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 PaddleLite+NNAdapter+HuaweiAscendNPU for amd64 and arm64 的部署库

  - For amd64

    - full_publish 编译

      ```shell
      $ ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_huawei_ascend_npu=ON --nnadapter_huawei_ascend_npu_sdk_root=/usr/local/Ascend/ascend-toolkit/latest full_publish
      ```

    - 替换头文件和库

      ```shell
      # 清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      # 替换 include 目录
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      # 替换 NNAdapter 运行时库
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/huawei_ascend_npu/
      # 替换 NNAdapter device HAL 库
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libhuawei_ascend_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/huawei_ascend_npu/
      # 替换 libpaddle_full_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      # 替换 libpaddle_light_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      ```

  - For arm64

    - full_publish 编译

      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --toolchain=gcc --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_huawei_ascend_npu=ON --nnadapter_huawei_ascend_npu_sdk_root=/usr/local/Ascend/ascend-toolkit/latest full_publish
      ```

    - 替换头文件和库

      ```shell
      # 清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/huawei_ascend_npu/
      # 替换 NNAdapter device HAL 库
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libhuawei_ascend_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/huawei_ascend_npu/
      # 替换 libpaddle_full_api_shared.so
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_light_api_shared.so
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

- 替换头文件后需要重新编译示例程序

## 高级特性

- Dynamic Shape

  某些模型（例如：CV 类的PPOCR、NLP 类的BERT、ERNIE等模型）能够支持不同的输入尺寸，因此，为了避免每次推理时因输入尺寸变化而导致的模型重新生成的问题，在推理前需要使用 set_nnadapter_dynamic_info 接口设置额外的 dynamic shape 信息。

  目前昇腾NPU支持 Dynamic Shape 特性，根据 CANN 版本的不同，提供下列两种方式的设置。

  - 通过档位方式支持 Dynamic Shape（CANN Version ≥ 3.3.0)

    CANN 早期版本实现的不是真正意义上的动态 shape，而是基于档位方式提供有限的模型输入 shape 范围。

    **使用方式：**假设模型有三个输入，输入名分别为 x1、x2 和 x3（模型实际输入名需用 Netron 可视化模型后查看），模型想在 64，128，192，224 四个档位下推理。

    ```c++
    // Run inference by using light api with MobileConfig
    paddle::lite_api::MobileConfig mobile_config;
    // dynamic info
    std::map<std::string, std::vector<std::vector<int64_t>>> dynamic_shape_info;
    dynamic_shape_info["x1"] = {{1,64,4096},{1,128,4096},{1,192,4096},{1,224,4096}};
    dynamic_shape_info["x2"] = {{1,64,4096},{1,128,4096},{1,192,4096},{1,224,4096}};
    dynamic_shape_info["x3"] = {{1,64,64},{1,128,128},{1,192,192},{1,224,224}};
    mobile_config.set_nnadapter_dynamic_shape_info(dynamic_shape_info);
    ```

  - 通过 Shape Range 方式支持 Dynamic Shape（CANN Version ≥ 5.1.1.RC1)（试用特性）

    在最新的 CANN 版本，昇腾提供了 Shape Range 特性，实现了更广泛意义上的动态 shape。但该特性还未成熟，调通的模型有限，开发者若有兴趣可自行尝试。

    **使用方式一：**假设模型有两个输入，输入名分别为 x1 和 x2 ，设置模型输入 shape 范围时，注意第一列为 shape 最小值，第二列为 shape 最大值，需在nnadapter_context_properties里设置HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE =true 开启 shape range 特性。

    下例表示 x1 输入的最后一个维度在[100,150]的范围变化，x2 输入的最后一个维度在[25,50]的范围变化。

    ```c++
    // Run inference by using light api with MobileConfig
    paddle::lite_api::MobileConfig mobile_config;
    // dynamic info
    std::map<std::string, std::vector<std::vector<int64_t>>> dynamic_shape_info;
    std::string nnadapter_context_properties = "HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE=true"
    dynamic_shape_info["x1"] = {{1,3,32,100},{1,3,32,150}};
    dynamic_shape_info["x2"] = {{1,3,25},{1,3,50}};
    mobile_config.set_nnadapter_dynamic_shape_info(dynamic_shape_info);
    mobile_config.set_nnadapter_context_properties(nnadapter_context_properties);
    ```

    **使用方式二：**假设模型有两个输入，输入名分别为 x1 和 x2 ，设置输入 shape 范围时，如果用户不想指定维度的取值，则可以将其设置为-1，表示此维度可以使用>=1的任意取值，需设置nnadapter_context_properties，开启DYNAMIC_SHAPE_RANGE特性。

    ```c++
    // Run inference by using light api with MobileConfig
    paddle::lite_api::MobileConfig mobile_config;
    // dynamic info
    std::map<std::string, std::vector<std::vector<int64_t>>> dynamic_shape_info;
    std::string nnadapter_context_properties = "HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE=true"
    dynamic_shape_info["x1"] = {{1,3,32,-1}};
    dynamic_shape_info["x2"] = {{1,3,-1};
    mobile_config.set_nnadapter_dynamic_shape_info(dynamic_shape_info);
    mobile_config.set_nnadapter_context_properties(nnadapter_context_properties);
    ```

- 混合精度

  支持量化模型的推理，要求模型必须是由 PaddleSlim 产出的量化模型，例如：[resnet50_int8_per_layer](http://paddlelite-demo.bj.bcebos.com/devices/generic/models/resnet50_int8_per_layer.tar.gz)、[mobilenet_v1_int8_224_per_layer](http://paddlelite-demo.bj.bcebos.com/devices/generic/models/mobilenet_v1_int8_224_per_layer.tar.gz)模型。

  

  **使用方式：**

  ```c++
  // Run inference by using full api with CxxConfig
  paddle::lite_api::CxxConfig cxx_config;
  std::string nnadapter_mixed_precision_quantization_config_path{"nnadapter_mixed_precision_quantization_config_path.txt"};
  // nnadapter_mixed_precision
  cxx_config.set_nnadapter_mixed_precision_quantization_config_path(nnadapter_mixed_precision_quantization_config_path);
  ```

  **nnadapter_mixed_precision_quantization_config_path.txt：** 该文件表示在全量化模型里，昇腾硬件无法支持量化的算子。

  目前昇腾硬件运行全量化模型时，除 conv2d 和 fc 算子可运行在 INT8 精度下外，其余算子均需运行在 FP16 或 FP32 精度上。

  ```shell
  softmax
  pool2d
  elementwise_add
  relu
  ```

- 高级参数

  - HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS：

    指定昇腾设备的 ID号，例如HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0,1,2,3或HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0。设备 ID 是指当前机器中可用的昇腾芯片 ID，例如 1 块 Atlas 300I 卡包含 4 颗 Ascend310 芯片，因此，可以将 HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS 设置为 0~3，而 1 块 Atlas 300I pro 卡只有一颗 Ascend 310P 芯片，因此只能设置为 0。

  - HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH

    开启 Profiling 功能，指定保存 Profiling 结果的路径。

  - HUAWEI_ASCEND_NPU_DUMP_MODEL_FILE_PATH

    开启保存昇腾 om 模型功能，指定存放 om 模型的路径。

  - HUAWEI_ASCEND_NPU_PRECISION_MODE

    设置网络模型的精度模式，[详见参数配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC1alpha003/infacldevg/atctool/atlasatc_16_0086.html)。

  - HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH

    当取值为HUAWEI_ASCEND_NPU_PRECISION_MODE=allow_mix_precision时，如果用户想要在内置优化策略基础上进行调整，自行指定哪些算子允许降精度，哪些算子不允许降精度，则需要指定HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH参数设置文件路径，[详见参数配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC1alpha003/infacldevg/atctool/atlasatc_16_0088.html)。

  - HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE

    设置网络模型中算子是高精度实现模式还是高性能实现模式，[详见参数配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC1alpha003/infacldevg/atctool/atlasatc_16_0089.html)。

  - HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE

    设置optype列表中算子的实现方式，该参数当前仅支持设置某个具体算子的实现方式，不支持设置多个算子，[详见参数配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC1alpha003/infacldevg/atctool/atlasatc_16_0090.html)。

  - HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT

    使能全局weight压缩，AICore支持Weight压缩功能，模型转换时通过使能该参数，可以对Weight进行数据压缩，在进行算子计算时，对Weight进行解压缩，从而达到减少带宽、提高性能的目的。

  - HUAWEI_ASCEND_NPU_AUTO_TUNE_MODE

    设置算子的自动调优模式：控制TBE算子编译时，是否对算子进行调优，以便在昇腾AI处理器上寻找最好的性能配置，[详见参数配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC1alpha003/infacldevg/atctool/atlasatc_16_0092.html)。

  - HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE

    使能开启 Dynamic Shape Range 功能。

  - HUAWEI_ASCEND_NPU_INITIAL_BUFFER_LENGTH_OF_DYNAMIC_SHAPE_RANGE

    当开启 Dynamic Shape Range 后，指定输入输出所分配的内存大小。若不指定则采用默认值4 * 3 * 1024 * 1024 bytes。

  **使用方式：**

  ```c++
  // Run inference by using light api with MobileConfig
  paddle::lite_api::MobileConfig mobile_config;
  // nnadapter_context_properties, 多个参数之间使用;进行分割
  std::string nnadapter_context_properties = "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0;HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE=high_precision;HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE=LayerNorm;HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT=true;"
  mobile_config.set_nnadapter_context_properties(nnadapter_context_properties);
  ```

  

## 其他说明

- 华为达芬奇架构的 NPU 内部大量采用 `float16` 进行运算，因此，预测结果会存在偏差，但大部分情况下精度不会有较大损失。
