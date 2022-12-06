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

#include "lite/kernels/host/set_value_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename D>
void SetValueCompute<D>::Run() {
  auto& param = *param_.get_mutable<param_t>();
#define SET_VALUE_WITH_TENSOR(__starts__, __ends__, __steps__) \
  if (param.ValueTensor != nullptr) {                          \
    SetTensorValueKernel<D>(param.Input,                       \
                            param.ValueTensor,                 \
                            __starts__,                        \
                            __ends__,                          \
                            __steps__,                         \
                            param.axes,                        \
                            param.decrease_axes,               \
                            param.none_axes,                   \
                            param.Out);                        \
    return;                                                    \
  }

#define SET_VALUE(__precision__, __starts__, __ends__, __steps__, __values__) \
  if (!__values__.empty()) {                                                  \
    SetValue<__precision__>(param.Input,                                      \
                            __starts__,                                       \
                            __ends__,                                         \
                            __steps__,                                        \
                            param.axes,                                       \
                            param.decrease_axes,                              \
                            param.none_axes,                                  \
                            param.shape,                                      \
                            __values__,                                       \
                            param.Out);                                       \
    return;                                                                   \
  }

  if (param.StartsTensorList.size() > 0) {
    auto starts = GetDataFromTensorList(param.StartsTensorList);
    if (param.EndsTensorList.size() > 0) {
      auto ends = GetDataFromTensorList(param.EndsTensorList);
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SET_VALUE_WITH_TENSOR(starts, ends, steps)
        SET_VALUE(float, starts, ends, steps, param.fp32_values)
        SET_VALUE(double, starts, ends, steps, param.fp64_values)
        SET_VALUE(int, starts, ends, steps, param.int32_values)
        SET_VALUE(int64_t, starts, ends, steps, param.int64_values)
        SET_VALUE(int, starts, ends, steps, param.bool_values)
      } else {
        SET_VALUE_WITH_TENSOR(starts, ends, param.steps)
        SET_VALUE(float, starts, ends, param.steps, param.fp32_values)
        SET_VALUE(double, starts, ends, param.steps, param.fp64_values)
        SET_VALUE(int, starts, ends, param.steps, param.int32_values)
        SET_VALUE(int64_t, starts, ends, param.steps, param.int64_values)
        SET_VALUE(int, starts, ends, param.steps, param.bool_values)
      }
    } else {
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SET_VALUE_WITH_TENSOR(starts, param.ends, steps)
        SET_VALUE(float, starts, param.ends, steps, param.fp32_values)
        SET_VALUE(double, starts, param.ends, steps, param.fp64_values)
        SET_VALUE(int, starts, param.ends, steps, param.int32_values)
        SET_VALUE(int64_t, starts, param.ends, steps, param.int64_values)
        SET_VALUE(int, starts, param.ends, steps, param.bool_values)
      } else {
        SET_VALUE_WITH_TENSOR(starts, param.ends, param.steps)
        SET_VALUE(float, starts, param.ends, param.steps, param.fp32_values)
        SET_VALUE(double, starts, param.ends, param.steps, param.fp64_values)
        SET_VALUE(int, starts, param.ends, param.steps, param.int32_values)
        SET_VALUE(int64_t, starts, param.ends, param.steps, param.int64_values)
        SET_VALUE(int, starts, param.ends, param.steps, param.bool_values)
      }
    }
  } else {
    if (param.EndsTensorList.size() > 0) {
      auto ends = GetDataFromTensorList(param.EndsTensorList);
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SET_VALUE_WITH_TENSOR(param.starts, ends, steps)
        SET_VALUE(float, param.starts, ends, steps, param.fp32_values)
        SET_VALUE(double, param.starts, ends, steps, param.fp64_values)
        SET_VALUE(int, param.starts, ends, steps, param.int32_values)
        SET_VALUE(int64_t, param.starts, ends, steps, param.int64_values)
        SET_VALUE(int, param.starts, ends, steps, param.bool_values)
      } else {
        SET_VALUE_WITH_TENSOR(param.starts, ends, param.steps)
        SET_VALUE(float, param.starts, ends, param.steps, param.fp32_values)
        SET_VALUE(double, param.starts, ends, param.steps, param.fp64_values)
        SET_VALUE(int, param.starts, ends, param.steps, param.int32_values)
        SET_VALUE(int64_t, param.starts, ends, param.steps, param.int64_values)
        SET_VALUE(int, param.starts, ends, param.steps, param.bool_values)
      }
    } else {
      if (param.StepsTensorList.size() > 0) {
        auto steps = GetDataFromTensorList(param.StepsTensorList);
        SET_VALUE_WITH_TENSOR(param.starts, param.ends, steps)
        SET_VALUE(float, param.starts, param.ends, steps, param.fp32_values)
        SET_VALUE(double, param.starts, param.ends, steps, param.fp64_values)
        SET_VALUE(int, param.starts, param.ends, steps, param.int32_values)
        SET_VALUE(int64_t, param.starts, param.ends, steps, param.int64_values)
        SET_VALUE(int, param.starts, param.ends, steps, param.bool_values)
      } else {
        SET_VALUE_WITH_TENSOR(param.starts, param.ends, param.steps)
        SET_VALUE(
            float, param.starts, param.ends, param.steps, param.fp32_values)
        SET_VALUE(
            double, param.starts, param.ends, param.steps, param.fp64_values)
        SET_VALUE(
            int, param.starts, param.ends, param.steps, param.int32_values)
        SET_VALUE(
            int64_t, param.starts, param.ends, param.steps, param.int64_values)
        SET_VALUE(int, param.starts, param.ends, param.steps, param.bool_values)
      }
    }
  }
#undef SET_VALUE_WITH_TENSOR
#undef SET_VALUE
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(set_value,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::SetValueCompute<float>,
                     fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(set_value,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::SetValueCompute<int>,
                     int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(set_value,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::SetValueCompute<int64_t>,
                     int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(set_value,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::SetValueCompute<int>,
                     bool)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

REGISTER_LITE_KERNEL(set_value,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::SetValueCompute<double>,
                     double)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP64))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StepsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP64))})
    .Finalize();
