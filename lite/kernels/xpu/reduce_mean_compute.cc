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
  auto& param = Param<operators::ReduceMeanParam>();
  auto& ctx = this->ctx_->As<XPUContext>();
  const float* input = param.X->data<float>();
  auto x_dims = param.X->dims();
  int x_rank = x_dims.size();
  float* output = param.Out->mutable_data<float>(TARGET(kXPU));
  auto reduce_dim = param.dim;

  std::vector<int> idims;
  for (int i = 0; i < x_rank; i++) {
    idims.push_back(x_dims[i]);
  }

  auto type = xdnn::ReduceOp::REDUCE_MEAN;
  int r = xdnn::reduce(ctx.GetRawContext(),
                       input,
                       output,
                       idims.data(),
                       x_rank,
                       reduce_dim.data(),
                       reduce_dim.size(),
                       type);
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
