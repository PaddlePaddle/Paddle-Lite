// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/increment_compute.h"
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void increment(const T* input, const int n, const T step, T* out) {
  for (int i = 0; i < n; i++) {
    out[i] = input[i] + step;
    LOG(INFO) << "index:  " << out[i];
  }
}

template <typename T, PrecisionType PType>
void IncrementCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::IncrementParam>();
  int total_num = param.X->numel();
  auto mem_size = param.X->memory_size();
  CHECK_EQ(total_num * sizeof(T), mem_size);
  Tensor tmp;
  tmp.Resize({total_num});
  auto* tmp_data = tmp.template mutable_data<T>(TARGET(kHost));
  TargetWrapperXPU::MemcpySync(
      tmp_data, param.X->raw_data(), mem_size, IoDirection::DtoH);
  T step = static_cast<T>(param.step);
  increment<T>(tmp_data, total_num, step, tmp_data);
  auto* out_data = param.Out->template mutable_data(TARGET(kXPU), mem_size);
  TargetWrapperXPU::MemcpySync(out_data, tmp_data, mem_size, IoDirection::HtoD);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using increment_float32 =
    paddle::lite::kernels::xpu::IncrementCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(increment, kXPU, kFloat, kAny, increment_float32, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using increment_int32 =
    paddle::lite::kernels::xpu::IncrementCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(increment, kXPU, kFloat, kAny, increment_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using increment_int64 =
    paddle::lite::kernels::xpu::IncrementCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(increment, kXPU, kFloat, kAny, increment_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
