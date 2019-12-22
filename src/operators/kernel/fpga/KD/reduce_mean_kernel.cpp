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

#ifdef REDUCE_MEAN_OP

#include "operators/kernel/reduce_mean_kernel.h"
#include "fpga/KD/pes/pooling_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool ReduceMeanKernel<FPGA, float>::Init(ReduceMeanParam<FPGA> *param) {
  param->Output()->mutable_data<half>();

  std::vector<int> dims = param->Dims();
  if ((dims.size() == 2) && (dims[0] == 2) && (dims[1] == 3)) {
    zynqmp::PoolingPE& pe = param->context().pe<zynqmp::PoolingPE>();
    zynqmp::PoolingParam& pool_param = pe.param();

    pool_param.input = param->InputX()->zynqmpTensor();
    pool_param.output = param->Output()->zynqmpTensor();
    pool_param.type = zynqmp::PoolingType::AVERAGE;
    pool_param.globalPooling = true;
    int input_width = pool_param.input->shape().width();
    int input_height = pool_param.input->shape().height();
    pool_param.kernelSize = {input_width, input_height};
    pool_param.strides = {1, 1};
    pool_param.paddings = {1, 1};

    pe.init();
    pe.apply();
  } 
 
  return true;
}

template <>
void ReduceMeanKernel<FPGA, float>::Compute(const ReduceMeanParam<FPGA> &param) {

  std::vector<int> dims = param.Dims();
  if ((dims.size() == 2) && (dims[0] == 2) && (dims[1] == 3)) {

    zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
    zynqmp::PoolingPE& pe = context.pe<zynqmp::PoolingPE>();
    pe.dispatch();
  }

#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("reduce_mean",
                                                  param.Output()->zynqmpTensor());
#endif
}

template class ReduceMeanKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
