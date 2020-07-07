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

#include "lite/kernels/cuda/fc_compute.h"

#include <string>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
struct FcTypeTraits;

template <>
struct FcTypeTraits<float> {
  typedef float4 Type;
};

template <typename T>
__global__ void AddBiasV4(const int num, const T* bias, T* data, int K) {
  CUDA_KERNEL_LOOP(index, num) {
    int bias_idx = index % K;
    const T bias_ptr = bias[bias_idx];
    const T in_ptr = data[index];
    T packed_val;
    packed_val.x = in_ptr.x + bias_ptr.x;
    packed_val.y = in_ptr.y + bias_ptr.y;
    packed_val.z = in_ptr.z + bias_ptr.z;
    packed_val.w = in_ptr.w + bias_ptr.w;
    data[index] = packed_val;
  }
}

template <typename T>
__global__ void AddBiasReluV4(const int num, const T* bias, T* data, int K) {
  CUDA_KERNEL_LOOP(index, num) {
    int bias_idx = index % K;
    const T bias_ptr = bias[bias_idx];
    const T in_ptr = data[index];
    T packed_val;
    packed_val.x = fmaxf(0.f, in_ptr.x + bias_ptr.x);
    packed_val.y = fmaxf(0.f, in_ptr.y + bias_ptr.y);
    packed_val.z = fmaxf(0.f, in_ptr.z + bias_ptr.z);
    packed_val.w = fmaxf(0.f, in_ptr.w + bias_ptr.w);
    data[index] = packed_val;
  }
}

template <typename T>
__global__ void AddBias(const int num, const T* bias, T* data) {
  int offset = blockIdx.x * num;

  for (int i = threadIdx.x; i < num; i += blockDim.x) {
    T temp;
#if __CUDA_ARCH__ >= 350
    temp = __ldg(data + offset + i) + __ldg(bias + i);
#else
    temp = data[offset + i] + bias[i];
#endif
    data[offset + i] = temp;
  }
}

template <typename T>
__global__ void AddBiasRelu(const int num, const T* bias, T* data) {
  int offset = blockIdx.x * num;

  for (int i = threadIdx.x; i < num; i += blockDim.x) {
    T temp;
#if __CUDA_ARCH__ >= 350
    temp = __ldg(data + offset + i) + __ldg(bias + i);
#else
    temp = data[offset + i] + bias[i];
#endif
    data[offset + i] = static_cast<int>(temp > 0) * temp;
  }
}

template <typename T, PrecisionType PType>
void FcCompute<T, PType>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
}

template <typename T, PrecisionType PType>
void FcCompute<T, PType>::Run() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto& param = this->template Param<param_t>();

  const auto* x_data = param.input->template data<T>();
  const auto* w_data = param.w->template data<T>();
  const auto* b_data = param.bias ? param.bias->template data<T>() : nullptr;

  auto out_vec = param.output->dims().Vectorize();
  out_vec.back() = param.w->dims()[1];
  param.output->Resize(out_vec);
  auto* out_data = param.output->template mutable_data<T>(TARGET(kCUDA));

  int in_num_col_dims = param.in_num_col_dims;

  int M = static_cast<int>(
      param.input->dims().Slice(0, param.in_num_col_dims).production());
  int K = static_cast<int>(
      param.input->dims()
          .Slice(param.in_num_col_dims, param.input->dims().size())
          .production());
  int K2 = static_cast<int>(param.w->dims()[0]);
  int N = static_cast<int>(param.w->dims()[1]);
  CHECK_EQ(K, K2) << "x_w must be equal with y_h";

  CHECK(gemm_impl_->init(false, false, M, N, K, &context));
  gemm_impl_->run(1.0f, 0.0f, x_data, w_data, out_data, &context);

  if (b_data == nullptr) {
    return;
  }

  std::string activation_type = param.activation_type;
  if (N % 4 == 0) {
    const int threads = 256;
    const int num = M * N / 4;
    const int blocks = (num + threads - 1) / threads;
    typedef typename FcTypeTraits<T>::Type trans_type;
    const auto* bias_ptr_v4 = reinterpret_cast<const trans_type*>(b_data);
    auto* data_ptr_v4 = reinterpret_cast<trans_type*>(out_data);
    if (activation_type == "relu") {
      AddBiasReluV4<trans_type><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    } else if (activation_type == "") {
      AddBiasV4<trans_type><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    } else {
      LOG(FATAL) << "not supported activation type: " << activation_type;
    }
  } else {
    const int threads = 256;
    const int blocks = M;
    if (activation_type == "relu") {
      AddBiasRelu<T><<<blocks, threads, 0, stream>>>(N, b_data, out_data);
    } else if (activation_type == "") {
      AddBias<T><<<blocks, threads, 0, stream>>>(N, b_data, out_data);
    } else {
      LOG(FATAL) << "not supported activation type: " << activation_type;
    }
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using FcFp32 = paddle::lite::kernels::cuda::FcCompute<float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(fc, kCUDA, kFloat, kNCHW, FcFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
