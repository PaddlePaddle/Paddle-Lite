// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/fpga/reduce_mean_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void ReduceMeanCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  param.Out->mutable_data<float16>();

  auto dims = param.dim;

  if ((dims.size() == 2) && (dims[0] == 2) && (dims[1] == 3)) {
    zynqmp::PoolingParam& pool_param = pe_.param();

    pool_param.input = param.X->ZynqTensor();
    pool_param.output = param.Out->ZynqTensor();

    pool_param.type = zynqmp::PoolingType::AVERAGE;
    pool_param.globalPooling = true;
    int input_width = pool_param.input->shape().width();
    int input_height = pool_param.input->shape().height();
    pool_param.kernelSize = {input_height, input_width};
    pool_param.strides = {1, 1};
    pool_param.paddings = {0, 0};

    pe_.init();
    pe_.apply();
  } else {
    throw "dim is not supported";
  }
}

void ReduceMeanCompute::Run() { pe_.dispatch(); }

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reduce_mean,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ReduceMeanCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
