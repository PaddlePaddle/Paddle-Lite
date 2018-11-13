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
#ifdef POOL_OP

#include "operators/kernel/pool_kernel.h"

class PoolingArgs;
namespace paddle_mobile {
namespace operators {

template <>
bool PoolKernel<FPGA, float>::Init(PoolParam<FPGA> *param) {
  auto *input = const_cast<Tensor *>(param->Input());
  auto input_ptr = input->data<float>();
  Tensor *output = param->Output();
  fpga::format_fp16_ofm(output);
  auto output_ptr = output->mutable_data<float>();
  vector<int> ksize = param->Ksize();
  vector<int> strides = param->Strides();
  vector<int> paddings = param->Paddings();
  std::string pooling_type = param->PoolingType();

  fpga::PoolingArgs poolArgs = {0};
  poolArgs.mode = pooling_type == "max" ? 0 : 1;  // max:0, avg:1
  poolArgs.kernel_reciprocal =
      fpga::fp32_2_fp16(float(1.0 / (ksize[0] * ksize[1])));  // NOLINT
  poolArgs.image.address = input_ptr;
  poolArgs.image.channels = (uint32_t)input->dims()[1];
  poolArgs.image.height = (uint32_t)input->dims()[2];
  poolArgs.image.width = (uint32_t)input->dims()[3];
  poolArgs.image.pad_height = (uint32_t)paddings[0];
  poolArgs.image.pad_width = (uint32_t)paddings[1];
  poolArgs.image.scale_address = input->scale;
  poolArgs.output.address = output_ptr;
  poolArgs.output.scale_address = output->scale;
  poolArgs.kernel.height = (uint32_t)ksize[0];
  poolArgs.kernel.width = (uint32_t)ksize[1];
  poolArgs.kernel.stride_h = (uint32_t)strides[0];
  poolArgs.kernel.stride_w = (uint32_t)strides[1];
  param->SetFpgaArgs(poolArgs);
  return true;
}

template <>
void PoolKernel<FPGA, float>::Compute(const PoolParam<FPGA> &param) {
  fpga::ComputeFpgaPool(param.FpgaArgs());
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
