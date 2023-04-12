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

#include "lite/kernels/xpu/shape_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
template <PrecisionType PType>
void ShapeCompute<PType>::Run() {
  auto& param = this->template Param<operators::ShapeParam>();
  int* output_data = param.Out->template mutable_data<int>();
  auto in_dims = param.X->dims();
  for (size_t i = 0; i < in_dims.size(); ++i) {
    output_data[i] = in_dims[i];
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(shape,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::ShapeCompute<PRECISION(kAny)>,
                     xpu_shape)
    .BindInput("Input",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .Finalize();

REGISTER_LITE_KERNEL(shape,
                     kXPU,
                     kInt8,
                     kAny,
                     paddle::lite::kernels::xpu::ShapeCompute<PRECISION(kInt8)>,
                     xpu_shape_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .Finalize();
