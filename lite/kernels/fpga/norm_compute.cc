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

#include "lite/kernels/fpga/norm_compute.h"
// #include "lite/backends/arm/math/funcs.h"


namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void NormCompute::PrepareForRun() {
  auto& param = this->Param<operators::NormParam>();
  param.Out->mutable_data<float16>();


  zynqmp::NormParam& norm_param = pe_.param();
  norm_param.input = param.X->ZynqTensor();
  norm_param.output = param.Out->ZynqTensor();
  norm_param.epsilon = param.epsilon;

  pe_.init();
  pe_.apply();
}

void NormCompute::Run() {
  pe_.dispatch();
  pe_.param().output->saveToFile("norm.txt", true);
  // auto& ctx = this->ctx_->template As<ARMContext>();
  // auto& param = this->Param<operators::NormParam>();

  // auto input_dims = param.X->dims();
  // int dim_size = param.X->dims().size();
  // auto axis = (param.axis < 0) ? param.axis + dim_size : param.axis;

  // const auto* x_data = param.X->data<float>();
  // auto* o_data = param.Out->mutable_data<float>();
  // int pre_n = input_dims.count(0, axis);
  // int post_n = input_dims.count(axis + 1, dim_size);
  // int n = input_dims[axis];
  // lite::arm::math::norm(x_data, pre_n, n, post_n, param.epsilon, o_data, &ctx);
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    norm, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::NormCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Norm", {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .Finalize();
