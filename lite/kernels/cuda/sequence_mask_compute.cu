// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/sequence_mask_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
__global__ void SequenceMaskKernel(T* dst,
                                   const int64_t* src,
                                   int count,
                                   int maxlen) {
  CUDA_KERNEL_LOOP(index, count) {
    int src_idx = index / maxlen;
    int inner_idx = index % maxlen;
    dst[index] = static_cast<T>(inner_idx < src[src_idx] ? 1 : 0);
  }
}

template <typename T>
__global__ void VecMaxKernel(const T* in_data, T* out, const int count) {
  extern __shared__ T cache[];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int cache_index = threadIdx.x;
  T tmp = -1;

  while (i < count) {
    if (in_data[i] > tmp) {
      tmp = in_data[i];
    }
    i += blockDim.x * gridDim.x;
  }
  cache[cache_index] = tmp;

  __syncthreads();

  // perform parallel reduction, blockDim.x must be 2^n
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (cache_index < ib && cache[cache_index + ib] > cache[cache_index]) {
      cache[cache_index] = cache[cache_index + ib];
    }

    __syncthreads();

    ib /= 2;
  }
  if (cache_index == 0) {
    out[blockIdx.x] = cache[0];
  }
}

template <typename T, PrecisionType Ptype>
void SequenceMaskCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const auto* x = param.X;
  const int64_t* x_data = x->template data<int64_t>();
  auto* y = param.Y;
  int maxlen = param.maxlen;

  if (param.MaxLenTensor) {
    auto* len_tensor_data = param.MaxLenTensor->template data<int32_t>();
    int32_t len_data{0};
    TargetWrapperCuda::MemcpySync(
        &len_data, len_tensor_data, sizeof(int32_t), IoDirection::DtoH);
    maxlen = len_data;
  }

  if (maxlen < 0) {
    // choose algorithm according to magic_num.
    const int magic_num = 256;
    std::vector<int64_t> h_max_data;
    if (x->numel() < magic_num) {
      h_max_data.resize(x->numel());
      TargetWrapperCuda::MemcpySync(h_max_data.data(),
                                    x_data,
                                    x->numel() * sizeof(int64_t),
                                    IoDirection::DtoH);
    } else {
      const int threads = 256;
      const int blocks = (x->numel() + threads - 1) / threads;
      max_tensor_.Resize({blocks});
      auto* max_data = max_tensor_.mutable_data<int64_t>(TARGET(kCUDA));
      VecMaxKernel<
          int64_t><<<blocks, threads, threads * sizeof(int64_t), stream>>>(
          x_data, max_data, x->numel());
      h_max_data.resize(blocks);
      TargetWrapperCuda::MemcpyAsync(h_max_data.data(),
                                     max_data,
                                     sizeof(int64_t) * blocks,
                                     IoDirection::DtoH,
                                     stream);
      TargetWrapperCuda::StreamSync(stream);
    }
    auto maxlen_iterator =
        std::max_element(h_max_data.begin(), h_max_data.end());
    maxlen = h_max_data[std::distance(h_max_data.begin(), maxlen_iterator)];
  }

  auto y_dim = x->dims().Vectorize();
  y_dim.push_back(maxlen);
  y->Resize(y_dim);
  const int count = y->numel();
  auto* dst_data = y->template mutable_data<T>(TARGET(kCUDA));
  if (param.out_dtype == 5) {
    SequenceMaskKernel<
        T><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
        dst_data, x_data, count, maxlen);
  } else {
    LOG(FATAL) << "not supported out_dtype: " << param.out_dtype;
  }
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SeqMaskFp32 =
    paddle::lite::kernels::cuda::SequenceMaskCompute<float, PRECISION(kFloat)>;

using SeqMaskFp16 =
    paddle::lite::kernels::cuda::SequenceMaskCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(sequence_mask, kCUDA, kFloat, kNCHW, SeqMaskFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindInput("MaxLenTensor",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt32))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_mask, kCUDA, kFP16, kNCHW, SeqMaskFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindInput("MaxLenTensor",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt32))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
