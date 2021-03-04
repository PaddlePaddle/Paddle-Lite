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

#include "lite/kernels/host/linspace_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename Tin, typename Tout>
static void LinspaceFunc(const operators::LinspaceParam& param) {
  const auto* start_tensor = param.Start;
  const auto* stop_tensor = param.Stop;
  const auto* num_tensor = param.Num;
  auto* out_tensor = param.Out;
  const Tout start = static_cast<Tout>(start_tensor->template data<Tin>()[0]);
  const Tout stop = static_cast<Tout>(stop_tensor->template data<Tin>()[0]);
  const int num = num_tensor->data<int>()[0];
  Tout* out_data = out_tensor->template mutable_data<Tout>();

  if (num > 1) {
    // step should be of double type for all types
    double step = (static_cast<double>(stop - start)) / (num - 1);
    int half_num = num / 2;
    for (int i = 0; i < num; ++i) {
      if (i < half_num) {
        out_data[i] = static_cast<Tout>(start + step * i);
      } else {
        out_data[i] = static_cast<Tout>(stop - step * (num - i - 1));
      }
    }
  } else {
    out_data[0] = static_cast<Tout>(start);
  }
}

template <typename T, PrecisionType PType>
void LinspaceCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::LinspaceParam>();
  switch (param.Out->precision()) {
    case PRECISION(kFloat):
      LinspaceFunc<T, float>(param);
      break;
    case PRECISION(kInt32):
      LinspaceFunc<T, int32_t>(param);
      break;
    default:
      LOG(FATAL) << "Linspace op unsupport output data type: "
                 << lite_api::PrecisionToStr(param.Out->precision());
  }
  return;
}
}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using linspace_float =
    paddle::lite::kernels::host::LinspaceCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(linspace, kHost, kFloat, kAny, linspace_float, float32)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Stop",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Num",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

using linspace_int32 =
    paddle::lite::kernels::host::LinspaceCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(linspace, kHost, kInt32, kAny, linspace_int32, int32)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Stop",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Num",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
