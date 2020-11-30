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

#include "lite/kernels/host/uniform_random_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void UniformRandomKernelFunctor(Tensor* out, float min, float max, int seed) {
  T* p_out_data = out->mutable_data<T>();
  int64_t size = out->numel();
  memset(p_out_data, 0, size * sizeof(T));
  unsigned int out_seed = static_cast<unsigned int>(seed);
  std::minstd_rand engine;
  if (out_seed == 0) {
    out_seed = std::random_device()();
  }
  engine.seed(out_seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  for (int64_t i = 0; i < size; ++i) {
    p_out_data[i] = dist(engine);
  }
}

void UniformRandomCompute::Run() {
  auto& param = this->template Param<param_t>();
  switch (param.dtype) {
    case static_cast<int>(lite::core::FluidType::FP64):
      UniformRandomKernelFunctor<double>(
          param.Out, param.min, param.max, param.seed);
      break;
    case static_cast<int>(lite::core::FluidType::FP32):
      UniformRandomKernelFunctor<float>(
          param.Out, param.min, param.max, param.seed);
      break;
    default:
      LOG(ERROR) << "Unsupported data type for uniform_random op:"
                 << param.dtype;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(uniform_random,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::UniformRandomCompute,
                     def)
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
