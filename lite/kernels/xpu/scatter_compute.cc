// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/scatter_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, typename TID>
void ScatterCompute<T, TID>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& x_dims = param.x->dims();
  auto& index_dims = param.indexs->dims();
  std::vector<TID> cpu_index(param.indexs->numel(), 0);
  TargetWrapperXPU::MemcpySync(cpu_index.data(),
                               param.indexs->template data<TID>(),
                               param.indexs->numel() * sizeof(TID),
                               IoDirection::DtoH);

  xdnn::VectorParam<TID> indices{
      cpu_index.data(),
      param.indexs->numel(),
      const_cast<TID*>(param.indexs->template data<TID>())};
  CHECK(index_dims.size() == 1 || index_dims.size() == 0 ||
        (index_dims.size() == 2 && index_dims[1] == 1))
      << "invalid index dim: " << index_dims;
  int64_t dim0 = x_dims[0];
  CHECK_GT(dim0, 0) << "invalid dim0";
  int64_t dim1 = x_dims.production() / dim0;
  int r = xdnn::copy<T>(ctx.GetRawContext(),
                        param.x->template data<T>(),
                        param.output->template mutable_data<T>(TARGET(kXPU)),
                        param.x->numel());
  CHECK_EQ(r, 0);
  r = xdnn::scatter<T, TID>(
      ctx.GetRawContext(),
      param.updates->template data<T>(),
      param.output->template mutable_data<T>(TARGET(kXPU)),
      indices,
      dim0,
      dim1,
      param.overwrite);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using ScatterCompute_fp32_int64 =
    paddle::lite::kernels::xpu::ScatterCompute<float, int64_t>;
using ScatterCompute_fp32_int32 =
    paddle::lite::kernels::xpu::ScatterCompute<float, int>;

REGISTER_LITE_KERNEL(
    scatter, kXPU, kFloat, kNCHW, ScatterCompute_fp32_int64, ids_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    scatter, kXPU, kFloat, kNCHW, ScatterCompute_fp32_int32, ids_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();
