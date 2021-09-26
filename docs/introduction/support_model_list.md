# 支持模型

目前，Paddle-Lite 已严格验证 52 个模型的精度和性能。对视觉类模型做到了充分的支持，覆盖分类、检测和定位，也包含了特色的 OCR 模型的支持。对 NLP 模型也做到了广泛支持，包含翻译、语义表达等等。

除了已严格验证的模型，Paddle-Lite 对其他 CV 和 NLP 模型也可以做到大概率支持。


| 类别 | 类别细分 | 模型 | 支持平台 |
|------|----|----|---|
| CV | 分类 | MobileNetV1 | ARM，X86，HuaweiKirinNPU，RockchipNPU，MediatekAPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | MobileNetV2 | ARM，X86，HuaweiKirinNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | DPN68 | ARM，X86 |
| CV | 分类 | AlexNet | ARM，X86 |
| CV | 分类 | DarkNet53 | ARM，X86 |
| CV | 分类 | DenseNet121 | ARM，X86 |
| CV | 分类 | EfficientNetB0 | ARM，X86 |
| CV | 分类 | GhostNet_x1_3 | ARM |
| CV | 分类 | HRNet_W18_C | ARM，X86 |
| CV | 分类 | RegNetX_4GF | ARM，X86 |
| CV | 分类 | Xception41 | ARM，X86 |
| CV | 分类 | ResNet18 | ARM，HuaweiKirinNPU，RockchipNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | ResNet50 | ARM，X86，HuaweiKirinNPU，RockchipNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | MnasNet| ARM，HuaweiKirinNPU，HuaweiAscendNPU |
| CV | 分类 | EfficientNetB0 | ARM，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | SqueezeNet | ARM，HuaweiKirinNPU，BaiduXPU |
| CV | 分类 | ShufflenetV2 | ARM，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | ShuffleNet | ARM |
| CV | 分类 | InceptionV4 | ARM，X86，HuaweiKirinNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | VGG16 | ARM，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | VGG19 | ARM，BaiduXPU，HuaweiAscendNPU|
| CV | 分类 | GoogleNet | ARM，X86，BaiduXPU |
| CV | 检测 | MobileNet-SSD | ARM，HuaweiKirinNPU#，HuaweiAscendNPU# |
| CV | 检测 | YOLOv3-MobileNetV3 | ARM，HuaweiKirinNPU#，HuaweiAscendNPU# |
| CV | 检测 | Faster RCNN | ARM |
| CV | 检测 | Mask RCNN# | ARM |
| CV | 检测 | yolov3_darknet53_270e_coco | ARM |
| CV | 检测 | ppyolo_2x | ARM |
| CV | 检测 | solov2_r50_fpn_1x | ARM |
| CV | 检测 | yolov3_mobilenet_v3 | ARM |
| CV | 检测 | yolov3_r50vd_dcn | ARM |
| CV | OCR | ch_ppocr_mobile_v2.0_cls_infer | ARM，X86 |
| CV | OCR | ch_ppocr_mobile_v2.0_det_infer | ARM |
| CV | OCR | ch_ppocr_mobile_v2.0_rec_infer | ARM |
| CV | OCR | ch_ppocr_server_v2.0_rec_infer | ARM |
| CV | OCR | CRNN | ARM |
| CV | OCR | DB | ARM |
| CV | OCR | OCR-Attention | ARM |
| CV | REG | inference_dnn | ARM |
| CV | 分割 | Deeplabv3 | ARM |
| CV | 分割 | UNet | ARM |
| CV | 分割 | bisenet | ARM |
| CV | 分割 | fastscnn | ARM |
| CV | 分割 | bisenet_v2 | ARM |
| CV | 人脸 | FaceDetection | ARM |
| CV | 人脸 | FaceBoxes#| ARM |
| CV | 人脸 | BlazeFace# | ARM |
| CV | 人脸 | MTCNN | ARM |
| CV | GAN | CycleGAN# | HuaweiKirinNPU |
| NLP | 机器翻译 | Transformer# | ARM，HuaweiKirinNPU# |
| NLP | 机器翻译 | BERT | BaiduXPU |
| NLP | 语义表示 | ERNIE | BaiduXPU |

**注意：** 


1. 模型列表中 `#` 代表该模型链接来自[ PaddlePaddle/models ](https://github.com/PaddlePaddle/models)，否则为推理模型的下载链接
2. 支持平台列表中 HuaweiKirinNPU# 代表 ARM + HuaweiKirinNPU 异构计算，否则为 HuaweiKirinNPU 计算
3. 支持平台列表中 HuaweiAscendNPU# 代表 X86 或 ARM+HuaweiAscendNPU 异构计算，否则为 HuaweiAscendNPU 计算
