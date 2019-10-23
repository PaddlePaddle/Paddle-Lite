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

#include "lite/kernels/fpga/gru_compute.h"
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/gru_utils.h"
#include "lite/backends/arm/math/sequence2batch.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

inline lite_api::ActivationType GetGruActType(const std::string& type) {
  if (type == "sigmoid") {
    return lite_api::ActivationType::kSigmoid;
  } else if (type == "tanh") {
    return lite_api::ActivationType::kTanh;
  } else if (type == "relu") {
    return lite_api::ActivationType::kRelu;
  } else if (type == "identity") {
    return lite_api::ActivationType::kIndentity;
  } else {
    LOG(FATAL) << "unsupported activation type: " << type;
  }
}

void GRUCompute::Run() {
  auto& param = this->Param<param_t>();
  param.hidden->mutable_data<float16>();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    gru, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::GRUCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
