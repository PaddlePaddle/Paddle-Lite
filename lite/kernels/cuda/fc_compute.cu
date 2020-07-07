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
__global__ void AddBiasV2(const int num, const T* bias, T* data, int K) {
  CUDA_KERNEL_LOOP(index, num) {
    int bias_idx = index % K;
    const T bias_ptr = bias[bias_idx];
    const T in_ptr = data[index];
    T packed_val;
    packed_val.x = in_ptr.x + bias_ptr.x;
    packed_val.y = in_ptr.y + bias_ptr.y;
    data[index] = packed_val;
  }
}

template <>
__global__ void AddBiasV2(const int num,
                          const half2* bias,
                          half2* data,
                          int K) {
  CUDA_KERNEL_LOOP(index, num) {
    int bias_idx = index % K;
    const half2 bias_ptr = bias[bias_idx];
    const half2 in_ptr = data[index];
#if __CUDA_ARCH__ >= 530
    data[index] = __hadd2(in_ptr, bias_ptr);
#else
    half2 packed_val;
    packed_val.x = __hadd(in_ptr.x, bias_ptr.x);
    packed_val.y = __hadd(in_ptr.y, bias_ptr.y);
    data[index] = packed_val;
#endif
  }
}

template <typename T>
__global__ void AddBiasReluV2(const int num, const T* bias, T* data, int K) {
  CUDA_KERNEL_LOOP(index, num) {
    int bias_idx = index % K;
    const T bias_ptr = bias[bias_idx];
    const T in_ptr = data[index];
    T packed_val;
    packed_val.x = fmaxf(0.f, in_ptr.x + bias_ptr.x);
    packed_val.y = fmaxf(0.f, in_ptr.y + bias_ptr.y);
    data[index] = packed_val;
  }
}

template <>
__global__ void AddBiasReluV2(const int num,
                              const half2* bias,
                              half2* data,
                              int K) {
  CUDA_KERNEL_LOOP(index, num) {
    int bias_idx = index % K;
    const half2 bias_ptr = bias[bias_idx];
    const half2 in_ptr = data[index];
#if __CUDA_ARCH__ >= 530
    data[index] = __hmul2(__hgt2(in_ptr + bias_ptr, __float2half2_rn(0.f)),
                          in_ptr + bias_ptr);
#else
    const float2 bias = __half22float2(bias_ptr);
    const float2 in = __half22float2(in_ptr);
    data[index] = __floats2half2_rn(
        bias.x + in.x > 0.0f ? static_cast<float>(bias.x + in.x) : 0.0f,
        bias.y + in.y > 0.0f ? static_cast<float>(bias.y + in.y) : 0.0f);
#endif
  }
}

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

template <>
__global__ void AddBias(const int num, const half* bias, half* data) {
  int offset = blockIdx.x * num;

  for (int i = threadIdx.x; i < num; i += blockDim.x) {
    half temp;
#if __CUDA_ARCH__ >= 350
    temp = __hadd(__ldg(data + offset + i), __ldg(bias + i));
#else
    temp = __hadd(data[offset + i], bias[i]);
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

template <>
__global__ void AddBiasRelu<half>(const int num, const half* bias, half* data) {
  int offset = blockIdx.x * num;

  for (int i = threadIdx.x; i < num; i += blockDim.x) {
    half temp;
#if __CUDA_ARCH__ >= 350
    temp = __hadd(__ldg(data + offset + i), __ldg(bias + i));
#else
    temp = __hadd(data[offset + i], bias[i]);
#endif

#if __CUDA_ARCH__ >= 530
    data[offset + i] =
        __hgt(temp, __float2half(0.0f)) ? temp : __float2half(0.0f);
#else
    data[offset + i] =
        __float2half(__half2float(temp) > 0.f ? __half2float(temp) : 0.f);
#endif
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

template <>
void FcCompute<half, PRECISION(kFP16)>::Run() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto& param = this->template Param<param_t>();

  const auto* x_data = param.input->template data<half>();
  const auto* w_data = param.w->template data<half>();
  const auto* b_data = param.bias ? param.bias->template data<half>() : nullptr;

  auto out_vec = param.output->dims().Vectorize();
  out_vec.back() = param.w->dims()[1];
  param.output->Resize(out_vec);
  auto* out_data = param.output->template mutable_data<half>(TARGET(kCUDA));

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
  if (N % 2 == 0) {
    const int threads = 256;
    const int num = M * N / 2;
    const int blocks = (num + threads - 1) / threads;
    const auto* bias_ptr_v2 = reinterpret_cast<const half2*>(b_data);
    auto* data_ptr_v2 = reinterpret_cast<half2*>(out_data);
    if (activation_type == "relu") {
      AddBiasReluV2<half2><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v2, data_ptr_v2, N / 2);
    } else if (activation_type == "") {
      AddBiasV2<half2><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v2, data_ptr_v2, N / 2);
    } else {
      LOG(FATAL) << "not supported activation type: " << activation_type;
    }
  } else {
    const int threads = 256;
    const int blocks = M;
    if (activation_type == "relu") {
      AddBiasRelu<half><<<blocks, threads, 0, stream>>>(N, b_data, out_data);
    } else if (activation_type == "") {
      AddBias<half><<<blocks, threads, 0, stream>>>(N, b_data, out_data);
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

using FcFp16 = paddle::lite::kernels::cuda::FcCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(fc, kCUDA, kFloat, kNCHW, FcFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kCUDA, kFP16, kNCHW, FcFp16, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
