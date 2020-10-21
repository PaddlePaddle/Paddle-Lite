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

#include <algorithm>
#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {

class ConcatPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(false);
    output->setDataLocation(CPU);
    bool cacheable = true;
    for (auto in : param_.inputs) {
      cacheable &= in->cacheable();
    }
    output->setCacheable(cacheable);
    return true;
  }

  void apply() {}

  void concat2D() {
    int offset = 0;
    float16* out_data = param_.output->data<float16>();
    for (unsigned int n = 0; n < param_.inputs.size(); n++) {
      Tensor* input = param_.inputs[n];
      Shape& input_shape = input->shape();

      float16* src = input->data<float16>();
      memcpy(out_data + offset, src, input_shape.numel() * sizeof(float16));
      offset += input_shape.numel();
    }
    Tensor* output = param_.output;
    output->flush();
  }

  void concat3D() {
    auto input = param_.inputs;
    Tensor* output = param_.output;
    int axis = param_.axis;
    int num = input.size();
    int rows = 1;
    auto dim_0 = input[0]->shape().dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int out_rows = rows, out_cols = 0;

    std::vector<int64_t> input_cols(input.size());
    for (int i = 0; i < num; ++i) {
      int t_cols = input[i]->shape().numel() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }

    // computation
    for (int k = 0; k < out_rows; ++k) {
      float16* dst_ptr = output->data<float16>() + k * out_cols;
      int col_idx = 0;
      for (int j = 0; j < num; ++j) {
        int col_len = input_cols[j];
        const float16* src_prt = input[j]->data<float16>() + k * col_len;
        memcpy(dst_ptr + col_idx, src_prt, sizeof(float16) * col_len);
        col_idx += col_len;
      }
    }
    output->flush();
  }

  bool dispatch() {
    Tensor* output = param_.output;
    Shape& output_shape = output->shape();

    float scale = 0;
    for (unsigned int n = 0; n < param_.inputs.size(); n++) {
      Tensor* input = param_.inputs[n];
      input->syncToCPU();
      input->unalignImage();
      scale = std::max(scale, input->scale()[0]);
    }
    output->scale()[0] = scale;
    output->scale()[1] = 1.0f / scale;

    if (output_shape.dimSize() == 3) {
      concat3D();
      return true;
    }

    if (output_shape.dimSize() == 2) {
      concat2D();
      return true;
    }

    float16* out_data = param_.output->data<float16>();
    int channel_sum = 0;
    int out_channel = output_shape.channel();
    for (unsigned int n = 0; n < param_.inputs.size(); n++) {
      Tensor* input = param_.inputs[n];
      Shape& input_shape = input->shape();
      int wh = output_shape.width() * output_shape.height();
      for (int j = 0; j < wh; j++) {
        float16* src = input->data<float16>() + j * input_shape.channel();
        memcpy(out_data + j * out_channel + channel_sum,
               src,
               input_shape.channel() * sizeof(float16));
      }
      channel_sum += input_shape.channel();
    }
    output->flush();
    return true;
  }

  ConcatParam& param() { return param_; }

 private:
  ConcatParam param_;
};

}  // namespace zynqmp
}  // namespace paddle
