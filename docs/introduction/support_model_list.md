# 支持模型

目前已严格验证24个模型的精度和性能，对视觉类模型做到了较为充分的支持，覆盖分类、检测和定位，包含了特色的OCR模型的支持，并在不断丰富中。

| 类别 | 类别细分 | 模型 | 支持Int8 | 支持平台 |
|-|-|:-:|:-:|-:|
| CV  | 分类 | mobilenetv1 | Y | arm，x86，npu，rknpu，apu |
| CV  | 分类 | mobilenetv2 | Y | arm，x86，npu |
| CV  | 分类 | resnet18 | Y | arm，npu |
| CV  | 分类 | resnet50 | Y | arm，x86，npu，xpu |
| CV  | 分类 | mnasnet |  | arm，npu |
| CV  | 分类 | efficientnet |  | arm |
| CV  | 分类 | squeezenetv1.1 |  | arm，npu |
| CV  | 分类 | ShufflenetV2 | Y | arm |
| CV  | 分类 | shufflenet | Y | arm |
| CV  | 分类 | inceptionv4 | Y | arm，x86，npu |
| CV  | 分类 | vgg16 | Y | arm |
| CV  | 分类 | googlenet | Y  | arm，x86 |
| CV  | 检测 | mobilenet_ssd | Y | arm，npu* |
| CV  | 检测 | mobilenet_yolov3 | Y | arm，npu* |
| CV | 检测 | Faster RCNN |  | arm |
| CV | 检测 | Mask RCNN |  | arm |
| CV | 分割 | Deeplabv3 | Y | arm |
| CV  | 分割 | unet |  | arm |
| CV  | 人脸 | facedetection |  | arm |
| CV  | 人脸 | facebox |  | arm |
| CV  | 人脸 | blazeface | Y | arm |
| CV  | 人脸 | mtcnn |  | arm |
| CV  | OCR | ocr_attention |  | arm |
| NLP  | 机器翻译 | transformer |  | arm，npu* |

> **注意：** npu* 代表ARM+NPU异构计算
