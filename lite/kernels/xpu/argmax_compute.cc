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

#include "lite/kernels/xpu/argmax_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ArgmaxCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto x = param.X;
  auto out = param.Out;
  int axis = param.Axis;
  std::vector<int> x_dims(x->dims().data().begin(), x->dims().data().end());
  int rank = x_dims.size();
  if (axis < 0) {
    axis += rank;
  }
  if (param.dtype == -1 || param.dtype == 3) {
    // default int64_t, static_cast<int>(lite::core::FluidType::INT64) == 3
    int r =
        xdnn::argmax<float, int64_t>(ctx.GetRawContext(),
                                     x->data<float>(),
                                     out->mutable_data<int64_t>(TARGET(kXPU)),
                                     x_dims,
                                     axis);
    CHECK_EQ(r, 0);
  } else {
    // else if (param.dtype == 2) {}    xpu not support int32 now
    LOG(FATAL) << "argmax unsupported param type for xpu: " << param.dtype;
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(arg_max,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ArgmaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
