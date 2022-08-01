// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/meshgrid_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void MeshgridCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::MeshgridParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  std::vector<lite::Tensor*>& ins = param.X;
  std::vector<lite::Tensor*>& outs = param.Out;
  int64_t size = ins.size();

  std::vector<const T*> x_list;
  std::vector<std::vector<int>> x_shape_list;
  for (int i = 0; i < size; ++i) {
    std::vector<int> x_shape(1);
    switch (ins[i]->dims().size()) {
      case 0:
        x_shape[0] = 1;
        break;
      case 1:
        x_shape[0] = ins[i]->dims()[0];
        break;
      default:
        LOG(FATAL) << "Meshgrid Op expected scalar or 1D tensor in the input "
                      "tensor list";
        break;
    }
    x_shape_list.push_back(x_shape);
    x_list.push_back(reinterpret_cast<const T*>(ins[i]->template data<T>()));
  }

  std::vector<T*> out_ptrs;
  for (auto out : outs) {
    out_ptrs.push_back(out->template mutable_data<T>(TARGET(kXPU)));
  }

  int r =
      xdnn::meshgrid<T>(ctx.GetRawContext(), x_list, out_ptrs, x_shape_list);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using meshgridFP32 =
    paddle::lite::kernels::xpu::MeshgridCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(meshgrid, kXPU, kFloat, kAny, meshgridFP32, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using meshgridFP16 =
    paddle::lite::kernels::xpu::MeshgridCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(meshgrid, kXPU, kFP16, kAny, meshgridFP16, float16)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kAny))})
    .Finalize();

using meshgridInt32 =
    paddle::lite::kernels::xpu::MeshgridCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(meshgrid, kXPU, kFloat, kAny, meshgridInt32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using meshgridInt64 =
    paddle::lite::kernels::xpu::MeshgridCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(meshgrid, kXPU, kFloat, kAny, meshgridInt64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
