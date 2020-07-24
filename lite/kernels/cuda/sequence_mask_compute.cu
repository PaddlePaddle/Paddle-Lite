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
    maxlen = static_cast<int>(
        thrust::reduce(thrust::device_pointer_cast(x_data),
                       thrust::device_pointer_cast(x_data) + x->numel(),
                       static_cast<int64_t>(0),
                       thrust::maximum<int64_t>()));
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
