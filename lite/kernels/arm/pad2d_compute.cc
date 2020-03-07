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

#include "lite/kernels/arm/pad2d_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void Pad2dCompute::Run() {
  auto& param = Param<operators::Pad2dParam>();
  const lite::Tensor* inputs = param.X;
  auto* out = param.Out;

  if (param.mode == "constant") {
    mode_ = 0;
  } else if (param.mode == "reflect") {
    mode_ = 1;
  } else if (param.mode == "edge") {
    mode_ = 2;
  } else {
    LOG(FATAL) << "Unknown mode type";
  }

  pad_h_ = {param.paddings[0], param.paddings[1]};
  pad_w_ = {param.paddings[2], param.paddings[3]};
  pad_value_ = param.pad_value;
  data_format_ = param.data_format;
  if (mode_ == 2) {
    // nchw
    auto input_dims = inputs->dims();
    CHECK_LE(pad_h_[0], input_dims[2] - 1)
        << "pad top size must <= inputs height - 1";
    CHECK_LE(pad_h_[1], input_dims[2] - 1)
        << "pad bottom size must <= inputs height - 1";
    CHECK_LE(pad_w_[0], input_dims[3] - 1)
        << "pad left size must <= inputs width - 1";
    CHECK_LE(pad_w_[1], input_dims[3] - 1)
        << "pad right size must  <= inputs width - 1";
  }
  lite::arm::math::pad2d_func(inputs, out, mode_, pad_h_, pad_w_, pad_value_);
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    pad2d, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::Pad2dCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
