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

#include "lite/kernels/xpu/concat_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType>
void ConcatCompute<InType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto ins = param.x;
  auto out = param.output;
  int64_t axis = param.axis < 0
                     ? param.axis + static_cast<int>(ins[0]->dims().size())
                     : param.axis;

  std::vector<const float*> x_list;
  std::vector<std::vector<int>> xdims_list;
  for (int i = 0; i < ins.size(); i++) {
    if (ins[i]->numel() > 0) {
      xdims_list.push_back(std::vector<int>());
      for (int j = 0; j < ins[i]->dims().size(); j++) {
        xdims_list[i].push_back(ins[i]->dims()[j]);
        if (sizeof(InType) == 8) {
          xdims_list[i].back() = xdims_list[i].back() * 2;
        }
      }
      x_list.push_back(
          reinterpret_cast<const float*>(ins[i]->template data<InType>()));
    }
  }
  if (x_list.size() > 0) {
    int r = xdnn::concat<float>(
        ctx.GetRawContext(),
        x_list,
        reinterpret_cast<float*>(
            out->template mutable_data<InType>(TARGET(kXPU))),
        xdims_list,
        axis);

    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(concat,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ConcatCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(concat,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ConcatCompute<int>,
                     concat_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(concat,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ConcatCompute<int64_t>,
                     concat_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
