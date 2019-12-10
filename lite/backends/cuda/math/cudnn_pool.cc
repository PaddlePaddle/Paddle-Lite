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

#include "lite/backends/cuda/math/cudnn_pool.h"
#include "lite/backends/cuda/math/activation.h"
#include "lite/backends/cuda/math/scale.h"
#include "lite/backends/cuda/math/type_trans.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

inline void UpdatePadding(std::vector<int>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::vector<int>& data_dims,
                          const std::vector<int>& strides,
                          const std::vector<int>& ksize) {
  if (paddings->size() == data_dims.size()) {
    for (size_t i = 0; i < data_dims.size(); ++i) {
      int copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    CHECK(data_dims.size() * 2 == paddings->size())
        << "Paddings size should be the same or twice as the pooling size.";
  }
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

inline void UpdateKsize(std::vector<int>* ksize,
                        const std::vector<int>& data_dims) {
  ksize->resize(static_cast<size_t>(data_dims.size()));
  for (size_t i = 0; i < ksize->size(); ++i) {
    *(ksize->begin() + i) = static_cast<int>(data_dims[i]);
  }
}

template <>
bool CudnnPool2DNHWC<PRECISION(kFloat)>::create(
    const operators::PoolParam& param, Context<TARGET(kCUDA)>* ctx) {
  return true;
}

template <>
bool CudnnPool2DNHWC<PRECISION(kFloat)>::init(const operators::PoolParam& param,
                                              Context<TARGET(kCUDA)>* ctx) {
  this->stream_ = ctx->exec_stream();
  CUDNN_CHECK(cudnnCreate(&this->handle_));
  CUDNN_CHECK(cudnnSetStream(this->handle_, this->stream_));

  cudnnCreateTensorDescriptor(&this->input_desc_);
  cudnnCreateTensorDescriptor(&this->output_desc_);
  cudnnCreatePoolingDescriptor(&this->pooling_desc_);

  return create(param, ctx);
}

template <>
bool CudnnPool2DNHWC<PRECISION(kFloat)>::run(
    const operators::PoolParam& param) {
  auto x_dims = param.x->dims();
  auto o_dims = param.output->dims();
  int batch = x_dims[0];
  const float* in_data = param.x->data<float>();
  float* out_data = param.output->mutable_data<float>(TARGET(kCUDA));

  int ih = x_dims[1];
  int iw = x_dims[2];  // nchw
  int ic = x_dims[3];

  int oh = o_dims[1];
  int ow = o_dims[2];
  int oc = o_dims[3];

  std::vector<int> ksize = param.ksize;
  std::vector<int> strides = param.strides;
  std::vector<int> paddings = *(param.paddings.get());

  std::string pooling_type = param.pooling_type;
  bool global_pooling = param.global_pooling;
  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;

  std::vector<int> data_dims = {ih, iw};
  UpdatePadding(&paddings, global_pooling, adaptive, data_dims, strides, ksize);

  if (data_dims.size() * 2 == paddings.size()) {
    for (size_t i = 0; i < data_dims.size(); ++i) {
      paddings.erase(paddings.begin() + i + 1);
    }
  }

  if (global_pooling) {
    UpdateKsize(&ksize, data_dims);
  }
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->input_desc_,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         batch,
                                         ic,
                                         ih,
                                         iw));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->output_desc_,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         batch,
                                         oc,
                                         oh,
                                         ow));
  cudnnPoolingMode_t mode;
  if (pooling_type == "max") {
    mode = CUDNN_POOLING_MAX;
  } else {
    mode = exclusive ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                     : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  }
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(this->pooling_desc_,
                                          mode,
                                          CUDNN_NOT_PROPAGATE_NAN,
                                          ksize.size(),
                                          ksize.data(),
                                          paddings.data(),
                                          strides.data()));
  float alpha = 1.0f;
  float beta = 0.0f;
  CUDNN_CHECK(cudnnPoolingForward(this->handle_,
                                  this->pooling_desc_,
                                  &alpha,
                                  this->input_desc_,
                                  in_data,
                                  &beta,
                                  this->output_desc_,
                                  out_data));

  return true;
}

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
