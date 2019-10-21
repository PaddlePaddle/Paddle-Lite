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

#include "lite/utils/cv/image_preprocess.h"
#include <arm_neon.h>
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
// init
ImagePreprocess::ImagePreprocess(const uint8_t* src,
                                 ImageFormat srcFormat,
                                 TransParam param) {
  this->srcFormat_ = srcFormat;
  this->transParam_ = param;
}
void ImagePreprocess::imageCovert(const uint8_t* src,
                                  uint8_t* dst,
                                  ImageFormat srcFormat,
                                  ImageFormat dstFormat) {
  paddle::lite::utils::cv::ImageConvert img_convert;
  img_convert.choose(src,
                     dst,
                     srcFormat,
                     dstFormat,
                     this->transParam_.iw,
                     this->transParam_.ih);
}
void ImagePreprocess::image2Tensor(const uint8_t* src,
                                   Tensor* dstTensor,
                                   LayOut layout,
                                   float* means,
                                   float* scales) {
  paddle::lite::utils::cv::Image2Tensor img2tensor;
  img2tensor.choose(src,
                    dstTensor,
                    this->srcFormat_,
                    layout,
                    this->transParam_.iw,
                    this->transParam_.ih,
                    means,
                    scales);
}
void ImagePreprocess::imageTransform(const uint8_t* src, uint8_t* dst) {
  std::vector<Transform> v_trans = this->transParam_.v_trans;
  paddle::lite::utils::cv::ImageTransform img_trans;
  for (auto val : v_trans) {
    if (val == Resize) {
      img_trans.resize(src,
                       dst,
                       this->srcFormat_,
                       this->transParam_.iw,
                       this->transParam_.ih,
                       this->transParam_.ow,
                       this->transParam_.oh);
    } else if (val == Flip) {
      img_trans.flip(src,
                     dst,
                     this->srcFormat_,
                     this->transParam_.iw,
                     this->transParam_.ih,
                     this->transParam_.flip_param);
    } else if (val == Rotate) {
      img_trans.rotate(src,
                       dst,
                       this->srcFormat_,
                       this->transParam_.iw,
                       this->transParam_.ih,
                       this->transParam_.rotate_param);
    } else {
      printf("val: %d does not support !", val);
    }
  }
}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
