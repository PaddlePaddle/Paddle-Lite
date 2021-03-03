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
#include "lite/kernels/fpga/activation_compute.h"

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void FcCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  if (param.enable_int8) {
    input_max_ = zynqmp::float_to_half(127 * param.input_scale);
  }
  // ====================================================
  zynqmp::FullyConnectedParam& fc_param = pe_.param();

  param.output->mutable_data<float16>();
  fc_param.input = param.input->ZynqTensor();
  fc_param.output = param.output->ZynqTensor();
  fc_param.filter = param.w->ZynqTensor();
  fc_param.bias = param.bias->ZynqTensor();
  fc_param.bias->flush();

  if (activation_map.count(param.activation_type)) {
    fc_param.activeParam.type = activation_map[param.activation_type];
  }

  pe_.init();
  pe_.apply();
}

void FcCompute::Run() {
  auto& param = this->Param<param_t>();
  if (param.enable_int8) {
    param.input->ZynqTensor()->max()[0] = input_max_;
  }
  pe_.dispatch();

#ifdef FPGA_PRINT_TENSOR
  zynqmp::FullyConnectedParam& fc_param = pe_.param();
  Debugger::get_instance().registerOutput("fc", fc_param.output);
#endif
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
