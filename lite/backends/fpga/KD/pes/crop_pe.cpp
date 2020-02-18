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

#include "lite/backends/fpga/KD/pes/crop_pe.hpp"

namespace paddle {
namespace zynqmp {

bool CropPE::dispatch() {
  Tensor* input = param_.input;
  input->syncToCPU();
  const auto axis = param_.axis;
  std::vector<int> shape = param_.shape;
  auto* out = param_.output;

  Shape out_shape = out->shape();
  float16* src_ptr = reinterpret_cast<float16*>(input->data<float16>());
  float16* dst_ptr = reinterpret_cast<float16*>(
      out->mutableData<float16>(DataType::FP16, out_shape));

  std::vector<int> offsets = param_.offsets;

  int input_c = input->shape().channel();
  int input_h = input->shape().height();
  int input_w = input->shape().width();

  int out_c = out->shape().channel();
  int out_h = out->shape().height();
  int out_w = out->shape().width();
  if (axis == 1) {
    int index = 0;

    int offset_h = offsets[0];
    int offset_w = offsets[0];
    int offset_c = offsets[0];

    if (offsets.size() == 3) {
      offset_h = offsets[1];
      offset_w = offsets[2];
      offset_c = offsets[0];
    }

    for (int h = 0; h < out_h; h++) {
      for (int w = 0; w < out_w; w++) {
        float16* crop_start = src_ptr + (h + offset_h) * input_w * input_c +
                              (offset_w * input_c) + offset_c;
        std::memcpy(dst_ptr + h * (out_w * out_c) + w * out_c,
                    crop_start,
                    out_c * sizeof(float16));
      }
    }
  } else if (axis == 2) {
    int offset_h = offsets[0];
    int offset_w = offsets[0];

    if (offsets.size() == 2) {
      offset_h = offsets[0];
      offset_w = offsets[1];
    }

    for (int h = 0; h < out_h; h++) {
      float16* crop_start =
          src_ptr + (h + offset_h) * input_w * input_c + (offset_w * input_c);
      std::memcpy(dst_ptr + h * out_w * input_c,
                  crop_start,
                  out_w * input_c * sizeof(float16));
    }
  }
  out->flush();
  out->copyScaleFrom(input);
  return true;
}

}  // namespace zynqmp
}  // namespace paddle
