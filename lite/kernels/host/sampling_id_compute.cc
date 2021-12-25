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

#include "lite/kernels/host/sampling_id_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void SamplingIdCompute<T>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  int seed = param.seed;

  auto engine_ = std::make_shared<std::mt19937_64>();
  if (seed == 0) {
    std::random_device rd;
    seed = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  }
  engine_->seed(seed);
}

template <class T>
void SamplingIdCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  const lite::Tensor* x = param.x;
  lite::Tensor* out = param.out;

  int64_t batch_size = x->dims()[0];
  int64_t width = x->dims()[1];
  auto x_data = x->data<T>();
  auto out_data = out->mutable_data<int64_t>();
  std::uniform_real_distribution<T> dist(static_cast<T>(param.min),
                                         static_cast<T>(param.max));

  for (int64_t i = 0; i < batch_size; ++i) {
    T r = dist(*engine_);
    int64_t idx = width - 1;
    for (int64_t j = 0; j < width; ++j) {
      if ((r -= x_data[i * width + j]) < 0) {
        idx = j;
        break;
      }
    }
    out_data[i] = idx;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using sampling_id_float = paddle::lite::kernels::host::SamplingIdCompute<float>;
REGISTER_LITE_KERNEL(sampling_id, kHost, kAny, kAny, sampling_id_float, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
