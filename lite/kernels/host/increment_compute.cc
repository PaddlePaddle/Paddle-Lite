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

#include "lite/kernels/host/increment_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void increment(const T* input, const int n, const T step, T* out) {
  for (int i = 0; i < n; i++) {
    out[i] = input[i] + step;
  }
}

void IncrementCompute::Run() {
  auto& param = this->Param<operators::IncrementParam>();

  int total_num = param.X->numel();
  switch (param.X->precision()) {
    case PRECISION(kFloat): {
      const auto* x_data = param.X->data<float>();
      auto* o_data = param.Out->mutable_data<float>();
      float step = static_cast<float>(param.step);
      increment(x_data, total_num, step, o_data);
      break;
    }
    case PRECISION(kInt64): {
      const auto* x_data = param.X->data<int64_t>();
      auto* o_data = param.Out->mutable_data<int64_t>();
      int64_t step = static_cast<int64_t>(param.step);
      increment(x_data, total_num, step, o_data);
      break;
    }
    case PRECISION(kInt32): {
      const auto* x_data = param.X->data<int32_t>();
      auto* o_data = param.Out->mutable_data<int32_t>();
      int32_t step = static_cast<int32_t>(param.step);
      increment(x_data, total_num, step, o_data);
      break;
    }
    default:
      LOG(FATAL) << "unsupport input type "
                 << PrecisionToStr(param.X->precision());
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(increment,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::IncrementCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
