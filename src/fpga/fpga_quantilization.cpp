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

#include "fpga/fpga_quantilization.h"
#include <algorithm>

namespace paddle_mobile {
namespace fpga {

template <typename Dtype>
static void chw_to_hwc(Dtype* data_in, Dtype* data_out, int num, int channel,
                       int height, int width) {
  int offset_height = 0;

  for (int n = 0; n < num; n++) {
    int amount_per_row = width * channel;
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        int offset_height = h * amount_per_row;
        for (int w = 0; w < width; w++) {
          *(data_out + offset_height + w * channel + c) = *(data_in++);
        }
      }
    }
    data_out += num;
  }
}

template <typename Dtype>
static Dtype find_max(Dtype* data, int num) {
  Dtype max = 0;
  for (int i = 0; i < num; ++i) {
    max = std::max(max, data[i]);
  }
  return max;
}

// template <typename Dtype>
void quantify_filter(framework::Tensor* filter) {
  DLOG << "quantilize_filter........";

  float scale = 0;
  float fix_range = static_cast<float>((1 << (8 - 1)) - 1);

  const int batch_size = filter->dims()[0];
  const int channel = filter->dims()[1];
  const int height = filter->dims()[2];
  const int width = filter->dims()[3];

  int8_t* int_data = nullptr;
  int8_t* tmp_data = new int8_t[filter->numel()];

  // 32bit filter -> 8bit filter;
  if (filter->type() == typeid(float)) {
    float* float_data = filter->data<float>();
    float max = find_max<float>(float_data, filter->numel());

    scale = (max / fix_range);

    for (int i = 0; i < filter->numel(); ++i) {
      tmp_data[i] = (int8_t)float_data[i] * scale;
    }
    int_data = filter->mutable_data<int8_t>();
  } else {
    int8_t max = find_max<int8_t>(filter->data<int8_t>(), filter->numel());
    scale = (max / fix_range);

    for (int i = 0; i < filter->numel(); ++i) {
      tmp_data[i] = filter->data<int8_t>()[i];
    }
    int_data = filter->mutable_data<int8_t>();
  }
  // NCHW -> NHWC;
  chw_to_hwc<int8_t>(tmp_data, int_data, batch_size, channel, height, width);
  delete tmp_data;
  *(filter->fpga_args().scale_pointer()) = scale;
}

}  // namespace fpga
}  // namespace paddle_mobile
