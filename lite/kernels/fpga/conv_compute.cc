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

#include "lite/kernels/fpga/conv_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void ConvCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  // ====================================================
  zynqmp::ConvParam& conv_param = pe_.param();
  // auto& param = Param<operators::ConvParam>();

  input_.share_from_tensorlite(*param.x);
  output_.share_from_tensorlite(*param.output);
  filter_.share_from_tensorlite(*param.filter);

  conv_param.input = &input_;
  conv_param.output = &output_;
  conv_param.filter = &filter_;
  conv_param.groups = param.groups;
  conv_param.strides = param.strides;
  conv_param.paddings = param.paddings;
  conv_param.dilations = param.dilations;

  // std::vector<int> kernelSize;
  pe_.init();
  pe_.apply();
}

void ConvCompute::Run() {
  auto& param = this->Param<param_t>();
  pe_.dispatch();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    conv2d, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::ConvCompute, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
