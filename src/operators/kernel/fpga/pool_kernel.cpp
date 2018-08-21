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
  const Tensor *input = param->Input();
  auto input_ptr = input->data<half>();
  Tensor *output = param->Output();
  auto output_ptr = output->mutable_data<half>();
  vector<int> ksize = param->Ksize();
  vector<int> strides = param->Strides();
  vector<int> paddings = param->Paddings();

  fpga::PoolingArgs poolArgs;
  poolArgs.image.address = (void *)input_ptr;
  poolArgs.image.channels = input->dims()[1];
  poolArgs.image.height = input->dims()[2];
  poolArgs.image.width = input->dims()[3];
  poolArgs.image.pad_height = paddings[0];
  poolArgs.image.pad_width = paddings[1];
  poolArgs.output.address = output_ptr;
  poolArgs.kernel.height = ksize[0];
  poolArgs.kernel.width = ksize[1];
  poolArgs.kernel.stride_h = strides[0];
  poolArgs.kernel.stride_w = strides[1];
  param->SetFpgaArgs(poolArgs);
  return true;
}

template <>
void PoolKernel<FPGA, float>::Compute(const PoolParam<FPGA> &param) const {
#ifdef PADDLE_MOBILE_FPGA
  fpga::ComputeFpgaPool(param.FpgaArgs());
#endif
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
