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

#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/leaky_relu_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
__global__ void LeakyReluKernel(const int num,
                                const T alpha,
                                const T* input,
                                T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
#if __CUDA_ARCH__ >= 350
    output[index] = __ldg(input + index) >= 0 ? __ldg(input + index)
                                              : __ldg(input + index) * alpha;
#else
    output[index] = input[index] >= 0 ? input[index] : input[index] * alpha;
#endif
  }
}

void LeakyReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  int num = static_cast<int>(param.X->numel());
  float alpha = param.Leaky_relu_alpha;
  auto input = param.X->data<float>();
  auto output = param.Out->mutable_data<float>(TARGET(kCUDA));

  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  LeakyReluKernel<<<blocks, threads, 0, stream>>>(num, alpha, input, output);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(leaky_relu,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::LeakyReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .SetVersion("1.5.0")
    .BindPaddleOpVersion("leaky_relu", 1)
    .Finalize();
