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

#include "lite/kernels/fpga/elementwise_compute.h"
#include <string>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void ElementwiseAddCompute::PrepareForRun() {
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  auto& param = Param<operators::ElementwiseParam>();

  param.Out->mutable_data<float16>();

  ew_param.inputs = {param.X->ZynqTensor(), param.Y->ZynqTensor()};
  ew_param.output = param.Out->ZynqTensor();
  ew_param.axis = param.axis;
  ew_param.relu.enabled = false;

  pe_.init();
  pe_.apply();
}
void ElementwiseAddCompute::Run() { pe_.dispatch(); }

void ElementwiseAddActivationCompute::PrepareForRun() {
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  if (param.act_type != "relu") {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
  param.Out->mutable_data<float16>();
  ew_param.inputs = {param.X->ZynqTensor(), param.Y->ZynqTensor()};
  ew_param.output = param.Out->ZynqTensor();
  ew_param.axis = param.axis;
  ew_param.relu.enabled = true;
  pe_.init();
  pe_.apply();
}
void ElementwiseAddActivationCompute::Run() { pe_.dispatch(); }

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ElementwiseAddCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_add_activation,
    kFPGA,
    kFP16,
    kNHWC,
    paddle::lite::kernels::fpga::ElementwiseAddActivationCompute,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
