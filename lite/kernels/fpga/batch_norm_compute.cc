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

#include "lite/kernels/fpga/batch_norm_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void BatchNormCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  param.y->mutable_data<float16>();

  zynqmp::BatchnormParam& bn_param = pe_.param();
  bn_param.input = param.x->ZynqTensor();
  bn_param.output = param.y->ZynqTensor();
  bn_param.bias = param.bias->ZynqTensor();
  bn_param.scale = param.scale->ZynqTensor();
  bn_param.mean = param.mean->ZynqTensor();
  bn_param.variance = param.variance->ZynqTensor();
  bn_param.epsilon = param.epsilon;
  bn_param.activeParam.type = zynqmp::TYPE_NONE;

  pe_.init();
  pe_.apply();
}

void BatchNormCompute::Run() {
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::BatchnormParam& bn_param = pe_.param();
  Debugger::get_instance().registerOutput("batch_norm", bn_param.Y);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::BatchNormCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
