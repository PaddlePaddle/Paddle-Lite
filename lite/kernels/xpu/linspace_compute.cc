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

#include "lite/kernels/xpu/linspace_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void LinspaceCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::LinspaceParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const auto* start_tensor = param.Start;
  const auto* stop_tensor = param.Stop;
  const auto* num_tensor = param.Num;
  auto* out_tensor = param.Out;
  int64_t num = static_cast<int64_t>(num_tensor->template data<int>()[0]);
  int r = -1;
  switch (param.Out->precision()) {
    case PRECISION(kFloat):
      r = xdnn::linspace<T>(ctx.GetRawContext(),
                            out_tensor->template mutable_data<T>(TARGET(kXPU)),
                            static_cast<T>(start_tensor->template data<T>()[0]),
                            static_cast<T>(stop_tensor->template data<T>()[0]),
                            num);
      CHECK_EQ(r, 0);
      break;
    case PRECISION(kInt32):
      r = xdnn::linspace<T>(ctx.GetRawContext(),
                            out_tensor->template mutable_data<T>(TARGET(kXPU)),
                            static_cast<T>(start_tensor->template data<T>()[0]),
                            static_cast<T>(stop_tensor->template data<T>()[0]),
                            num);
      CHECK_EQ(r, 0);
      break;
    default:
      LOG(FATAL) << "Linspace op unsupport output data type: "
                 << lite_api::PrecisionToStr(param.Out->precision());
  }
  return;
}
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using linspace_float =
    paddle::lite::kernels::xpu::LinspaceCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(linspace, kXPU, kFloat, kAny, linspace_float, float32)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Stop",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Num",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using linspace_int32 =
    paddle::lite::kernels::xpu::LinspaceCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(linspace, kXPU, kInt32, kAny, linspace_int32, int32)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Stop",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Num",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
