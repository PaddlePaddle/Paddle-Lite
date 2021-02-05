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

#include "lite/kernels/xpu/reduce_mean_compute.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ReduceMeanCompute::Run() {
  auto& param = Param<operators::ReduceParam>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x_dims = param.X->dims();
  std::vector<int> x_shape;
  for (size_t i = 0; i < x_dims.size(); i++) {
    x_shape.push_back(static_cast<int>(x_dims[i]));
  }
  int r = xdnn::reduce_mean<float>(ctx.GetRawContext(),
                                   param.X->data<float>(),
                                   param.Out->mutable_data<float>(TARGET(kXPU)),
                                   x_shape,
                                   param.dim);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reduce_mean,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReduceMeanCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
