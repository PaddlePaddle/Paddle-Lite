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
#include "fpga/KD/pes/pooling_pe.hpp"

class PoolingArgs;
namespace paddle_mobile {
namespace operators {

template <>
bool PoolKernel<FPGA, float>::Init(PoolParam<FPGA>* param) {
  param->Output()->mutable_data<half>();

  zynqmp::PoolingPE& pe = param->context().pe<zynqmp::PoolingPE>();
  zynqmp::PoolingParam& pool_param = pe.param();

  pool_param.input = param->Input()->zynqmpTensor();
  pool_param.output = param->Output()->zynqmpTensor();
  pool_param.type = param->PoolingType() == "max"
                        ? zynqmp::PoolingType::MAX
                        : zynqmp::PoolingType::AVERAGE;
  pool_param.globalPooling = param->isGlobalPooling();
  pool_param.kernelSize = param->Ksize();
  pool_param.strides = param->Strides();
  pool_param.paddings = param->Paddings();

  pe.init();
  pe.apply();
  return true;
}

template <>
void PoolKernel<FPGA, float>::Compute(const PoolParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::PoolingPE& pe = context.pe<zynqmp::PoolingPE>();
  pe.dispatch();

  // param.Output()->zynqmpTensor()->printScale();
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
