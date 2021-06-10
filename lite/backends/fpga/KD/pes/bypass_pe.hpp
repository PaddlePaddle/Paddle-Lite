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

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
namespace paddle {
namespace zynqmp {

class BypassPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  bool dispatch() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;

    int count = input->aligned() ? input->shape().alignedElementCount()
                                 : input->shape().numel();
    BypassArgs args;
    args.input_data_type =
        input->dataType() == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    args.output_data_type =
        output->dataType() == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    args.input_layout_type = LAYOUT_HWC;
    args.output_layout_type = LAYOUT_HWC;
    args.image = {.address = input->data<void>(),
                  .scale_address = input->max(),
                  .channels = (uint32_t)count,
                  .width = 1,
                  .height = 1,
                  .pad_width = 0u,
                  .pad_height = 0u};
    args.output = {
        .address = output->data<void>(), .scale_address = output->max(),
    };
    input->syncToDevice();
    size_t aligned_remainder = count % 16;
    if (aligned_remainder > 0) {
      size_t dtype_size = CellSize(input->dataType());
      void* dst = input->data<char>() + input->shape().numel() * dtype_size;
      memset(dst, 0, aligned_remainder * dtype_size);
      fpga_flush(dst, aligned_remainder * dtype_size);
    }
    input->syncToDevice();
    input->invalidate();
    perform_bypass(args);
    output->invalidate();

    return true;
  }

  BypassParam& param() { return param_; }

 private:
  BypassParam param_;
};
}  // namespace zynqmp
}  // namespace paddle
