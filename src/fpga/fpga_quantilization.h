/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include "common/types.h"
#include "framework/lod_tensor.h"
#include "framework/tensor.h"

namespace paddle_mobile {

template <typename Dtype>
framework::Tensor* quantilize_filter(framework::Tensor* filter) {
  float scale = 0;
  // 32bit filter -> 8bit filter;
  float min = 0f;
  float max = 0f;
  if (filter->type() == typeid(float)) {
    float* floatData = originalFilter->data<float>();
    for (int i = 0; i < filter->numel(); ++i) {
      min = std::min(min, floatData[i]);
      max = std::max(max, floatData[i]);
    }

    float fix_range = (float)((1 << (8 - 1)) - 1);
    float float_range = max;
    scale = (float_range / fix_range);

    framework::Tensor* originalFilter = filter;
    framework::Tensor* quantFilter = new framework::Tensor();
    int8_t* intData = quantFilter->mutable_data<int8_t>();
    for (int i = 0; i < filter->numel(); ++i) {
      intData[i] = (int8_t)floatData[i] * scale;
    }
    quantFilter.scale = scale;
    // NCHW -> NHWC;
    return quantFilter;
  }
  return filter;
}

}  // namespace paddle_mobile
