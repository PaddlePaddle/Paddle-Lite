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

#include "lite/kernels/xpu/set_value_compute.h"
#include "lite/backends/xpu/target_wrapper.h"  // XPUScratchPadGuard
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SetValueCompute::PrepareForRun() {
#define OBTAIN_VALUE(__precision__, __data__)                                 \
  if (!__data__.empty()) {                                                    \
    int value_size = __data__.size();                                         \
    value_guard_ = TargetWrapperXPU::MallocScratchPad(value_size *            \
                                                      sizeof(__precision__)); \
    lite::TargetWrapperXPU::MemcpySync(value_guard_->addr_,                   \
                                       __data__.data(),                       \
                                       sizeof(__precision__) * value_size,    \
                                       IoDirection::HtoD);                    \
    return;                                                                   \
  }

  auto& param = this->template Param<param_t>();

  OBTAIN_VALUE(float, param.fp32_values)
  OBTAIN_VALUE(int, param.int32_values)
  OBTAIN_VALUE(int64_t, param.int64_values)
  OBTAIN_VALUE(float, param.fp16_values)
}

void SetValueCompute::SetValue(const std::vector<int64_t>& starts,
                               const std::vector<int64_t>& ends,
                               const std::vector<int64_t>& steps) {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
#define SET_VALUE(__precision__, __starts__, __ends__, __steps__, __data__) \
  if (!__data__.empty()) {                                                  \
    const __precision__* value_ptr =                                        \
        reinterpret_cast<__precision__*>(value_guard_->addr_);              \
    int r = xdnn::set_value<__precision__>(                                 \
        ctx.GetRawContext(),                                                \
        param.Input->template data<__precision__>(),                        \
        value_ptr,                                                          \
        param.Out->template mutable_data<__precision__>(TARGET(kXPU)),      \
        param.Input->dims().Vectorize(),                                    \
        {static_cast<int>(__data__.size())},                                \
        __starts__,                                                         \
        __ends__,                                                           \
        __steps__,                                                          \
        param.axes);                                                        \
    CHECK_EQ(r, 0);                                                         \
    return;                                                                 \
  }

#define SET_VALUE_WITH_TENSOR(                                         \
    __precision__, __starts__, __ends__, __steps__, __type__)          \
  if (param.ValueTensor != nullptr &&                                  \
      param.ValueTensor->precision() == __type__) {                    \
    int r = xdnn::set_value<__precision__>(                            \
        ctx.GetRawContext(),                                           \
        param.Input->template data<__precision__>(),                   \
        param.ValueTensor->template data<__precision__>(),             \
        param.Out->template mutable_data<__precision__>(TARGET(kXPU)), \
        param.Input->dims().Vectorize(),                               \
        param.ValueTensor->dims().Vectorize(),                         \
        __starts__,                                                    \
        __ends__,                                                      \
        __steps__,                                                     \
        param.axes,                                                    \
        {},                                                            \
        {});                                                           \
    CHECK_EQ(r, 0);                                                    \
    return;                                                            \
  }

  SET_VALUE_WITH_TENSOR(float, starts, ends, steps, PrecisionType::kFloat)
  SET_VALUE_WITH_TENSOR(int, starts, ends, steps, PrecisionType::kInt32)
  SET_VALUE_WITH_TENSOR(int64_t, starts, ends, steps, PrecisionType::kInt64)

  SET_VALUE(float, starts, ends, steps, param.fp32_values)
  SET_VALUE(int32_t, starts, ends, steps, param.int32_values)
  SET_VALUE(int64_t, starts, ends, steps, param.int64_values)
}

void SetValueCompute::Run() {
  auto& param = this->template Param<param_t>();

  auto set_value_x_shape = param.Input->dims().Vectorize();
  if (param.StartsTensorList.size() > 0) {
    auto starts = GetDataFromTensorList(param.StartsTensorList);
    if (param.EndsTensorList.size() > 0) {
      auto ends = GetDataFromTensorList(param.EndsTensorList);
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SetValue(starts, ends, steps);
      } else {
        SetValue(starts, ends, param.steps);
      }
    } else {
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SetValue(starts, param.ends, steps);
      } else {
        SetValue(starts, param.ends, param.steps);
      }
    }
  } else {
    if (param.EndsTensorList.size() > 0) {
      auto ends = GetDataFromTensorList(param.EndsTensorList);
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SetValue(param.starts, ends, steps);
      } else {
        SetValue(param.starts, ends, param.steps);
      }
    } else {
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SetValue(param.starts, param.ends, steps);
      } else {
        SetValue(param.starts, param.ends, param.steps);
      }
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(set_value,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::SetValueCompute,
                     DISABLE_XPU1FP32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(set_value,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::SetValueCompute,
                     DISABLE_XPU1Int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(set_value,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::SetValueCompute,
                     DISABLE_XPU1Int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
