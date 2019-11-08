// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "lite/api/paddle_api.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
#define PI 3.14159265f
#define Degrees2Radians(degrees) ((degrees) * (SK_ScalarPI / 180))
#define Radians2Degrees(radians) ((radians) * (180 / SK_ScalarPI))
#define ScalarNearlyZero (1.0f / (1 << 12))
typedef paddle::lite_api::Tensor Tensor;
// 颜色空间枚举
enum ImageFormat {
  RGBA = 0,
  BGRA,
  RGB,
  BGR,
  GRAY,
  NV21 = 11,
  NV12,
};
// 翻转枚举
enum FlipParam { X = 0, Y, XY };
// 输出tensor的格式枚举
enum LayOut { CHW = 0, HWC };
// 图像转换参数的结构体
typedef struct {
  int ih;                // 输入height
  int iw;                // 输入width
  int oh;                // 输出height
  int ow;                // 输出width
  FlipParam flip_param;  // 翻转参数
  float rotate_param;    // 旋转角度， 目前支持90, 180, 270
} TransParam;

class ImagePreprocess {
 public:
  /*
  * 图像处理函数初始化
  * param srcFormat: 输入图像的颜色空间
  * param dstFormat: 输出图像的颜色空间
  * param param: 输入图像的转换参数， 如输入图像大小、输出图像大小等
  */
  ImagePreprocess(ImageFormat srcFormat,
                  ImageFormat dstFormat,
                  TransParam param);

  /*
  * 颜色空间转换
  * 目前支持NV12/NV21_to_BGR(RGB), NV12/NV21_to_BGRA(RGBA),
  * BGR(RGB)和BGRA(RGBA)相互转换,
  * BGR(RGB)和RGB(BGR)相互转换,
  * BGR(RGB)和RGBA(BGRA)相互转换，以及BGR(RGB)和GRAY相互转换
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * 输入图像和输出图像的颜色空间参数从成员变量srcFormat_和dstFormat_获取
  */
  void imageCovert(const uint8_t* src, uint8_t* dst);
  /*
  * 颜色空间转换
  * 目前支持NV12/NV21_to_BGR(RGB), NV12/NV21_to_BGRA(RGBA),
  * BGR(RGB)和BGRA(RGBA)相互转换,
  * BGR(RGB)和RGB(BGR)相互转换,
  * BGR(RGB)和RGBA(BGRA)相互转换，以及BGR(RGB)和GRAY相互转换
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * param srcFormat: 输入图像的颜色空间,
  * 支持GRAY、NV12(NV21)、BGR(RGB)和BGRA(RGBA)
  * param dstFormat: 输出图像的颜色空间, 支持GRAY、BGR(RGB)和BGRA(RGBA)
  */
  void imageCovert(const uint8_t* src,
                   uint8_t* dst,
                   ImageFormat srcFormat,
                   ImageFormat dstFormat);
  /*
  * 图像resize
  * 颜色空间支持一通道（如GRAY）、NV12、 NV21、三通道（如BGR）和四通道（如BGRA）
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * 输入图像和输出图像大小数从成员变量transParam_获取
  */
  void imageResize(const uint8_t* src, uint8_t* dst);
  /*
  * 图像resize
  * 颜色空间支持一通道（如GRAY）、NV12、 NV21、三通道（如BGR）和四通道（如BGRA）
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * param srcFormat:
  * 输入图像的颜色空间，支持GRAY、NV12(NV21)、BGR(GRB)和BGRA(RGBA)格式
  * param srcw: 输入图像width
  * param srch: 输入图像height
  * param dstw: 输出图像width
  * param dsth: 输出图像height
  */
  void imageResize(const uint8_t* src,
                   uint8_t* dst,
                   ImageFormat srcFormat,
                   int srcw,
                   int srch,
                   int dstw,
                   int dsth);

  /*
  * 图像Rotate
  * 目前支持90, 180 和 270,
  * 颜色空间支持一通道（如GRAY）、三通道（如BGR）和四通道（如BGRA）
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * Rotate参数和图像大小参数从成员变量transParam_获取，图像大小取得是ow和oh
  */
  void imageRotate(const uint8_t* src, uint8_t* dst);
  /*
  * 图像Rotate
  * 目前支持90, 180 和 270,
  * 颜色空间支持一通道（如GRAY）、三通道（如BGR）和四通道（如BGRA）
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * param srcFormat: 输入图像的颜色空间，支持GRAY、BGR(GRB)和BGRA(RGBA)格式
  * param srcw: 输入图像width
  * param srch: 输入图像height
  * param degree: Rotate 度数
  */
  void imageRotate(const uint8_t* src,
                   uint8_t* dst,
                   ImageFormat srcFormat,
                   int srcw,
                   int srch,
                   float degree);
  /*
  * 图像Flip
  * 目前支持x、y 和 xy 翻转,
  * 颜色空间支持一通道（如GRAY）、三通道（如BGR）和四通道（如BGRA）
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * Flip参数和图像大小参数从成员变量transParam_获取，图像大小取得是ow和oh
  */
  void imageFlip(const uint8_t* src, uint8_t* dst);
  /*
  * 图像Flip
  * 目前支持x、y 和 xy 翻转,
  * 颜色空间支持一通道（如GRAY）、三通道（如BGR）和四通道（如BGRA）
  * param src: 输入图像数据
  * param dst: 输出图像数据
  * param srcFormat: 输入图像的颜色空间，支持GRAY、BGR(GRB)和BGRA(RGBA)格式
  * param srcw: 输入图像width
  * param srch: 输入图像height
  * param flip_param: Flip 参数
  */
  void imageFlip(const uint8_t* src,
                 uint8_t* dst,
                 ImageFormat srcFormat,
                 int srcw,
                 int srch,
                 FlipParam flip_param);
  /*
  * 将图像数据转换为tensor
  * 目前支持BGR（RGB）和BGRA（RGBA）数据转换为NCHW/NHWC格式的tensor
  * param src: 输入图像数据
  * param dstTensor: 输出Tensor数据
  * param layout: 输出Tensor格式，支持NCHW和NHWC两种格式
  * param means: 图像相应通道的均值
  * param scales: 图像相应通道的scale， 用于图像的归一化处理
  * 图像大小参数从成员变量transParam_获取，图像大小取得是ow和oh
  */
  void image2Tensor(const uint8_t* src,
                    Tensor* dstTensor,
                    LayOut layout,
                    float* means,
                    float* scales);
  /*
  * 将图像数据转换为tensor
  * 目前支持BGR（RGB）和BGRA（RGBA）数据转换为NCHW/NHWC格式的tensor
  * param src: 输入图像数据
  * param dstTensor: 输出Tensor数据
  * param srcFormat: 输入图像的颜色空间，支持BGR（RGB）和BGRA（RGBA）格式
  * param srcw: 输入图像width
  * param srch: 输入图像height
  * param layout: 输出Tensor格式，支持NCHW和NHWC两种格式
  * param means: 图像相应通道的均值
  * param scales: 图像相应通道的scale， 用于图像的归一化处理
  */
  void image2Tensor(const uint8_t* src,
                    Tensor* dstTensor,
                    ImageFormat srcFormat,
                    int srcw,
                    int srch,
                    LayOut layout,
                    float* means,
                    float* scales);

 private:
  ImageFormat srcFormat_;
  ImageFormat dstFormat_;
  TransParam transParam_;
};
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
