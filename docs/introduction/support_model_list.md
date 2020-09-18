# 支持模型

目前已严格验证28个模型的精度和性能，对视觉类模型做到了较为充分的支持，覆盖分类、检测和定位，包含了特色的OCR模型的支持，并在不断丰富中。

| 类别 | 类别细分 | 模型 | 支持平台 |
|-|-|:-|:-|
| CV | 分类 | mobilenetv1 | ARM，X86，NPU，RKNPU，APU |
| CV | 分类 | mobilenetv2 | ARM，X86，NPU |
| CV | 分类 | resnet18 | ARM，NPU |
| CV | 分类 | resnet50 | ARM，X86，NPU，XPU |
| CV | 分类 | mnasnet | ARM，NPU |
| CV | 分类 | efficientnet | ARM |
| CV | 分类 | squeezenetv1.1 | ARM，NPU |
| CV | 分类 | ShufflenetV2 | ARM |
| CV | 分类 | shufflenet | ARM |
| CV | 分类 | inceptionv4 | ARM，X86，NPU |
| CV | 分类 | vgg16 | ARM |
| CV | 分类 | vgg19 | XPU|
| CV | 分类 | googlenet | ARM，X86 |
| CV | 检测 | mobilenet_ssd | ARM，NPU* |
| CV | 检测 | mobilenet_yolov3 | ARM，NPU* |
| CV | 检测 | Faster RCNN | ARM |
| CV | 检测 | Mask RCNN | ARM |
| CV | 分割 | Deeplabv3 | ARM |
| CV | 分割 | unet | ARM |
| CV | 人脸 | facedetection | ARM |
| CV | 人脸 | facebox | ARM |
| CV | 人脸 | blazeface | ARM |
| CV | 人脸 | mtcnn | ARM |
| CV | OCR | ocr_attention | ARM |
| CV | GAN | CycleGAN | NPU |
| NLP | 机器翻译 | transformer | ARM，NPU* |
| NLP | 机器翻译 | BERT | XPU |
| NLP | 语义表示 | ERNIE | XPU |

**注意：** NPU* 代表ARM+NPU异构计算
