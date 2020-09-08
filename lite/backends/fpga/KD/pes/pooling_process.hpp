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

#ifndef pooling_process_hpp
#define pooling_process_hpp

#include <string.h>
#include <cmath>
#include <vector>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/tensor.hpp"

namespace paddle {
namespace zynqmp {

inline void pooling_split_channel(
    PoolingParam& param,                        // NOLINT
    std::vector<PoolingParam*>& splitParams) {  // NOLINT
  Tensor* input = param.input;
  Tensor* output = param.output;
  input->syncToCPU();

  int h_kernel = param.kernelSize[0];
  int w_kernel = param.kernelSize[1];
  if (param.globalPooling) {
    h_kernel = input->shape().height();
    w_kernel = input->shape().width();
  }

  int c = input->shape().channel();
  int w = input->shape().width();
  int wc_h_kernel = w * c * h_kernel;
  int dwconv_limit = 131072;
  int num = ceil(wc_h_kernel * 1.0f / dwconv_limit);

  while (input->shape().channel() % num != 0) {
    num++;
  }

  int channel = ceil(input->shape().channel() * 1.0f / num);

  float16* output_address = nullptr;
  float16* input_address = nullptr;
  float* out_scale_address = nullptr;

  for (int i = 0; i < num; i++) {
    PoolingParam* pooling_param = new PoolingParam();

    // input && output;
    Shape in_shape(
        NCHW, {1, channel, input->shape().height(), input->shape().width()});
    Shape out_shape(
        NCHW, {1, channel, output->shape().height(), output->shape().width()});
    if (num == 1) {
      pooling_param->input = input;
      pooling_param->output = output;
      input_address = input->data<float16>();
      output_address = output->data<float16>();
      out_scale_address = output->scale();
    } else {
      pooling_param->input = new Tensor();
      pooling_param->output = new Tensor();
      input_address =
          pooling_param->input->mutableData<float16>(FP16, in_shape);
      output_address =
          pooling_param->output->mutableData<float16>(FP16, out_shape);
      out_scale_address = pooling_param->output->scale();
    }

    PoolingArgs& args = pooling_param->poolingArgs;
    args.mode = param.type;
    args.kernel_reciprocal = fp32_2_fp16(1.0f / (w_kernel * h_kernel));
    if (param.globalPooling) {
      args.kernel_reciprocal = fp32_2_fp16(1.0f);
    }

    args.image.address = input_address;
    args.image.channels = channel;
    args.image.height = input->shape().height();
    args.image.width = input->shape().width();
    args.image.pad_height = param.paddings[0];
    args.image.pad_width = param.paddings[1];
    args.image.scale_address = input->scale();
    args.output.address = output_address;
    args.output.scale_address = out_scale_address;
    args.kernel.height = h_kernel;
    args.kernel.width = w_kernel;
    args.kernel.stride_h = param.strides[0];
    args.kernel.stride_w = param.strides[1];
    args.out_height = output->shape().height();
    args.out_width = output->shape().width();
    splitParams.push_back(pooling_param);
  }
}

}  // namespace zynqmp
}  // namespace paddle

#endif /* conv_process_hpp */
