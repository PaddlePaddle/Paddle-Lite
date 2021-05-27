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

#pragma once

#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
namespace paddle {
namespace zynqmp {

class SlicePE : public PE {
 public:
  bool init() { return true; }

  bool dispatch() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    std::vector<int> axes = param_.axes;
    std::vector<int32_t> starts = param_.starts;
    std::vector<int32_t> ends = param_.ends;
    std::vector<int> decrease_axis = param_.decrease_axis;
    std::vector<int> infer_flags = param_.infer_flags;

    if (axes.size() == 1 && axes[0] == 1) {
      int start = starts[0];
      int end = ends[0];

      int input_width = input->shape().width();
      int input_height = input->shape().height();
      int input_channel = input->shape().channel();
      int output_channel = output->shape().channel();
      float16* input_data = input->data<float16>();
      float16* output_data = output->data<float16>();

      int count = end - start;
      for (int i = 0; i < input_width * input_height; ++i) {
        memcpy(output_data + i * output_channel,
               input_data + i * input_channel + start,
               count * sizeof(float16));
      }
    } else if (axes.size() == 1 && axes[0] == 0) {
      int start = starts[0];
      int output_len = output->shape().numel();
      int* input_data = input->data<int32_t>();
      int* output_data = output->data<int32_t>();

      for (int i = 0; i < output_len; ++i) {
        output_data[i] = input_data[i + start];
      }
    }
    output->flush();

    return true;
  }

  SliceParam& param() { return param_; }

 private:
  SliceParam param_;
};

}  // namespace zynqmp
}  // namespace paddle
