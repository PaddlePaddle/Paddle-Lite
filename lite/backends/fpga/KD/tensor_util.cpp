/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>

#include "lite/backends/fpga/KD/tensor_util.hpp"

namespace paddle {
namespace zynqmp {

void chw_to_hwc(float* hwc_data,
                float* chw_data,
                int num,
                int channel,
                int height,
                int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          hwc_data[n * chw + h * wc + w * channel + c] = chw_data[index];
          index++;
        }
      }
    }
  }
}

void hwc_to_chw(float* chw_data,
                float* hwc_data,
                int num,
                int channel,
                int height,
                int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          chw_data[n * chw + c * wh + h * width + w] = hwc_data[index];
          index++;
        }
      }
    }
  }
}

float find_max(const Tensor& tensor) {
  float max = 0;
  Tensor& t = const_cast<Tensor&>(tensor);
  float* data = t.data<float>();
  for (int i = 0; i < t.shape().numel(); i++) {
    float value = data[i] > 0 ? data[i] : -data[i];
    max = std::max(value, max);
  }
  return max;
}

void hwc_to_chw(Tensor* src, Tensor* dst) {
  hwc_to_chw(dst->mutableData<float>(FP32, src->shape()),
             src->data<float>(),
             src->shape().num(),
             src->shape().channel(),
             src->shape().height(),
             src->shape().width());
}

void chw_to_hwc(Tensor* src, Tensor* dst) {
  chw_to_hwc(dst->mutableData<float>(FP32, src->shape()),
             src->data<float>(),
             src->shape().num(),
             src->shape().channel(),
             src->shape().height(),
             src->shape().width());
}

}  // namespace zynqmp
}  // namespace paddle
