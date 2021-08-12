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

#include "lite/kernels/xpu/logical_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <int LogicType>
void BinaryLogicalCompute<LogicType>::Run() {
  auto& param = this->template Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const auto* x = param.X->template data<bool>();
  const auto* y = param.Y->template data<bool>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = 0;
  switch (LogicType) {
    case LogicalType::LOGICAL_AND: {
      r = xdnn::logical_and<bool>(ctx.GetRawContext(), x, y, z, count);
      break;
    }
    default: {
      LOG(FATAL) << "LogicalType in logical_compute kernel "
                    "only supports logical_and[1] for xpu at this moment,"
                    "now it is "
                 << LogicType;
      break;
    }
  }
  CHECK_EQ(r, 0);
}

template <int LogicType>
void UnaryLogicalCompute<LogicType>::Run() {
  auto& param = this->template Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const auto* x = param.X->template data<bool>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = 0;
  switch (LogicType) {
    case LogicalType::LOGICAL_NOT: {
      r = xdnn::logical_not<bool>(ctx.GetRawContext(), x, z, count);
      break;
    }
    default: {
      LOG(FATAL) << "LogicalType in logical_compute kernel "
                    "only supports logical_not[0] for xpu at this moment,"
                    "now it is "
                 << LogicType;
      break;
    }
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(logical_and,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::BinaryLogicalCompute<
                         paddle::lite::kernels::xpu::LogicalType::LOGICAL_AND>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(logical_not,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::UnaryLogicalCompute<
                         paddle::lite::kernels::xpu::LogicalType::LOGICAL_NOT>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();
