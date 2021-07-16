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

#include "lite/kernels/xpu/correlation_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void CorrelationCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto* input1 = param.input1;
  auto* input2 = param.input2;
  auto* output = param.output;
  const int pad_size = param.pad_size;
  const int kernel_size = param.kernel_size;
  const int stride1 = param.stride1;
  const int stride2 = param.stride2;
  const int max_displacement = param.max_displacement;

  auto in_dims = input1->dims();
  auto* input1_data = input1->template data<T>();
  auto* input2_data = input2->template data<T>();
  T* out_data = output->template mutable_data<T>(TARGET(kXPU));

  int r = xdnn::correlation<T>(ctx.GetRawContext(),
                               input1_data,
                               input2_data,
                               out_data,
                               in_dims[0],
                               in_dims[1],
                               in_dims[2],
                               in_dims[3],
                               pad_size,
                               kernel_size,
                               stride1,
                               stride2,
                               max_displacement,
                               0);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(correlation,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::CorrelationCompute<float>,
                     def)
    .BindInput("Input1", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Input2", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
