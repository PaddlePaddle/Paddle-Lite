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

#include "lite/kernels/fpga/fc_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void FcCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  // ====================================================
  zynqmp::FullyConnectedParam& fc_param = pe_.param();

  input_.share_from_tensorlite(*param.input);
  output_.share_from_tensorlite(*param.output);
  filter_.share_from_tensorlite(*param.w);
  bias_.share_from_tensorlite(*param.bias);

  fc_param.input = &input_;
  fc_param.output = &output_;
  fc_param.filter = &filter_;
  fc_param.bias = &bias_;

  // std::vector<int> kernelSize;
  pe_.init();
  pe_.apply();
}

void FcCompute::Run() {
  auto& param = this->Param<param_t>();
  pe_.dispatch();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    fc, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::FcCompute, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
