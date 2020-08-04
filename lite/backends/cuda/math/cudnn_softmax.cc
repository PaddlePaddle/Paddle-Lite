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

#include "lite/backends/cuda/math/cudnn_softmax.h"

#include "lite/backends/cuda/math/cudnn_helper.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T, PrecisionType Ptype>
bool CudnnSoftmax<T, Ptype>::Init(const operators::SoftmaxParam& param,
                                  Context<TARGET(kCUDA)>* ctx) {
  this->stream_ = ctx->exec_stream();
  CUDNN_CHECK(cudnnCreate(&this->handle_));
  CUDNN_CHECK(cudnnSetStream(this->handle_, this->stream_));

  cudnnCreateTensorDescriptor(&this->bottom_desc_);
  cudnnCreateTensorDescriptor(&this->top_desc_);

  return Create(param, ctx);
}

template <typename T, PrecisionType Ptype>
bool CudnnSoftmax<T, Ptype>::Create(const operators::SoftmaxParam& param,
                                    Context<TARGET(kCUDA)>* ctx) {
  int axis = param.axis;
  if (axis < 0) {
    axis += param.x->dims().size();
  }
  int outer_num = param.x->dims().count(0, axis);
  int inner_num = param.x->dims().count(axis + 1, param.x->dims().size());

  int N = outer_num;
  int C = param.x->dims()[axis];
  int H = inner_num;
  int W = 1;

  const int stride_w = 1;
  const int stride_h = W * stride_w;
  const int stride_c = H * stride_h;
  const int stride_n = C * stride_c;
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc_,
                                           GetCudnnDataType<Ptype>(),
                                           N,
                                           C,
                                           H,
                                           W,
                                           stride_n,
                                           stride_c,
                                           stride_h,
                                           stride_w));
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc_,
                                           GetCudnnDataType<Ptype>(),
                                           N,
                                           C,
                                           H,
                                           W,
                                           stride_n,
                                           stride_c,
                                           stride_h,
                                           stride_w));
  handle_setup_ = true;
  return true;
}

template <typename T, PrecisionType Ptype>
bool CudnnSoftmax<T, Ptype>::Run(const operators::SoftmaxParam& param) {
  T* output_data = param.output->mutable_data<T>(TARGET(kCUDA));
  const T* input_data = param.x->data<T>();
  float alpha = 1.0f;
  float beta = 0.f;
  CUDNN_CHECK(cudnnSoftmaxForward(handle_,
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha,
                                  bottom_desc_,
                                  reinterpret_cast<const void*>(input_data),
                                  &beta,
                                  top_desc_,
                                  reinterpret_cast<void*>(output_data)));

  return true;
}

template class CudnnSoftmax<float, PRECISION(kFloat)>;
template class CudnnSoftmax<half, PRECISION(kFP16)>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
