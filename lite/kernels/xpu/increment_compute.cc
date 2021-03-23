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
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void IncrementXpu(const Tensor& x, const T step, Tensor* out, XPUContext* ctx) {
  const size_t x_size = static_cast<size_t>(x.numel());
  std::vector<T> data(x_size);
  TargetWrapperXPU::MemcpySync(
      &(data[0]), x.raw_data(), x_size * sizeof(T), IoDirection::DtoH);
  for (size_t i = 0; i < x_size; i++) {
    data[i] += step;
  }
  TargetWrapperXPU::MemcpySync(out->template mutable_data<T>(TARGET(kXPU)),
                               data.data(),
                               x_size * sizeof(T),
                               IoDirection::HtoD);
}

void IncrementCompute::Run() {
  auto& param = this->Param<operators::IncrementParam>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* x = param.X;
  auto* out = param.Out;
  switch (x->precision()) {
    case PRECISION(kFloat): {
      float step = static_cast<float>(param.step);
      IncrementXpu(*x, step, out, &ctx);
      break;
    }
    case PRECISION(kInt32): {
      int step = static_cast<int>(param.step);
      IncrementXpu(*x, step, out, &ctx);
      break;
    }
    case PRECISION(kInt64): {
      int64_t step = static_cast<int64_t>(param.step);
      IncrementXpu(*x, step, out, &ctx);
      break;
    }
    default:
      LOG(FATAL) << "unsupport input type: " << PrecisionToStr(x->precision());
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(increment,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::IncrementCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
