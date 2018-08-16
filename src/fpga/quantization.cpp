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

#include "fpga/quantization.h"
#include <algorithm>

namespace paddle_mobile {
namespace fpga {

template <typename Dtype>
static void chw_to_hwc(Dtype* data_in, Dtype* data_out, int64_t num,
                       int64_t channel, int64_t height, int64_t width) {
  for (int n = 0; n < num; n++) {
    int64_t amount_per_row = width * channel;
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        int64_t offset_height = h * amount_per_row;
        for (int w = 0; w < width; w++) {
          *(data_out + offset_height + w * channel + c) = *(data_in++);
        }
      }
    }
    data_out += num;
  }
}

template <typename Dtype>
static Dtype find_max(Dtype* data, int64_t num) {
  Dtype max = 0;
  for (int i = 0; i < num; ++i) {
    Dtype value = data[i];
    Dtype abs = value > 0 ? value : -value;
    max = std::max(max, abs);
  }
  return max;
}

// template <typename Dtype>
void quantize_filter(framework::Tensor *filter) {
  DLOG << "quantilize_filter........";

  float scale = 0;
  auto fix_range = static_cast<float>(std::pow(2, 8 - 1) - 1);

  const auto batch_size = filter->dims()[0];
  const auto channel = filter->dims()[1];
  const auto height = filter->dims()[2];
  const auto width = filter->dims()[3];

  auto* tmp_data = new int8_t[filter->numel()];

  // 32bit filter -> 8bit filter;
  if (filter->type() == typeid(float)) {
    auto* float_data = filter->data<float>();
    auto max = find_max<float>(float_data, filter->numel());

    scale = (fix_range / max);
    DLOG << "scale:" << scale;

    for (int i = 0; i < filter->numel(); ++i) {
      tmp_data[i] = (int8_t)(float_data[i] * scale);
    }
  } else {
    auto max = find_max<int8_t>(filter->data<int8_t>(), filter->numel());
    scale = (fix_range / max);
    std::memcpy(tmp_data, filter->data<int8_t>(), (size_t)filter->numel());
  }
  // NCHW -> NHWC;
  chw_to_hwc<int8_t>(tmp_data, filter->mutable_data<int8_t>(), batch_size,
                     channel, height, width);
  delete tmp_data;
  filter->SetFpgaScale(scale);
}

}  // namespace fpga
}  // namespace paddle_mobile
