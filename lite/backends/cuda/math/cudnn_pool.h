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

#pragma once
#include <cudnn.h>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <PrecisionType Ptype_out>
class CudnnPool2DBase {
 public:
  CudnnPool2DBase()
      : handle_(NULL),
        input_desc_(NULL),
        output_desc_(NULL),
        pooling_desc_(NULL) {}

  ~CudnnPool2DBase() {
    if (handle_ != NULL) {
      CUDNN_CHECK(cudnnDestroy(handle_));
    }
    if (input_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    }
    if (output_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
    }
    if (pooling_desc_) {
      cudnnDestroyPoolingDescriptor(pooling_desc_);
    }
  }

 protected:
  cudaStream_t stream_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
};

template <PrecisionType Ptype_out>
class CudnnPool2DNHWC : public CudnnPool2DBase<Ptype_out> {
 public:
  CudnnPool2DNHWC() : CudnnPool2DBase<Ptype_out>() {}
  virtual ~CudnnPool2DNHWC() = default;
  virtual bool init(const operators::PoolParam& param,
                    Context<TARGET(kCUDA)>* ctx);

  virtual bool create(const operators::PoolParam& param,
                      Context<TARGET(kCUDA)>* ctx);

  virtual bool run(const operators::PoolParam& param);
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
