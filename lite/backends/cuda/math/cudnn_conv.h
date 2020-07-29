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
class CudnnConv2DBase {
 public:
  CudnnConv2DBase()
      : handle_(NULL),
        fwd_algo_((cudnnConvolutionFwdAlgo_t)0),
        input_desc_(NULL),
        output_desc_(NULL),
        bias_desc_(NULL),
        filter_desc_(NULL),
        conv_desc_(NULL),
        act_desc_(NULL),
        workspace_data_(NULL),
        workspace_(NULL),
        workspace_fwd_sizes_(0),
        workspace_size_inbytes_(0) {}

  ~CudnnConv2DBase() {
    if (conv_desc_) {
      CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
    }
    if (input_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    }
    if (output_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
    }
    if (act_desc_) {
      CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc_));
    }
    if (bias_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
    }
    if (filter_desc_) {
      CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    }
    if (handle_ != NULL) {
      CUDNN_CHECK(cudnnDestroy(handle_));
    }
    ResetWorkSpace();
  }

 protected:
  void ResetWorkSpace() {
    if (workspace_data_ != NULL) {
      CUDA_CALL(cudaFree(workspace_data_));
    }
    workspace_data_ = NULL;
  }

 protected:
  cudaStream_t stream_;
  cudnnHandle_t handle_;
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionFwdAlgoPerf_t algo_perf_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;

  // activation descriptor
  cudnnActivationDescriptor_t act_desc_;
  bool with_relu_act_{true};

  void* workspace_data_;  // underlying storage
  void* workspace_;       // aliases into _workspaceData
  size_t workspace_fwd_sizes_;
  size_t workspace_size_inbytes_;  // size of underlying storage

  const bool use_tensor_core_ = true;
  const size_t workspace_limit_bytes_ = 4 * 1024 * 1024;

  // For int8
  Tensor temp_tensor_;
  Tensor scale_;
};

template <typename T, PrecisionType Ptype_out>
class CudnnConv2D : public CudnnConv2DBase<Ptype_out> {
 public:
  CudnnConv2D() : CudnnConv2DBase<Ptype_out>() {}
  virtual ~CudnnConv2D() = default;
  virtual bool init(const operators::ConvParam& param,
                    Context<TARGET(kCUDA)>* ctx);

  virtual bool create(const operators::ConvParam& param,
                      Context<TARGET(kCUDA)>* ctx);

  virtual bool run(const operators::ConvParam& param);
};

template <PrecisionType Ptype_out>
class CudnnConv2DInt8 : CudnnConv2DBase<Ptype_out> {
 public:
  CudnnConv2DInt8() : CudnnConv2DBase<Ptype_out>() {}
  virtual ~CudnnConv2DInt8() = default;
  virtual bool init(const operators::ConvParam& param,
                    Context<TARGET(kCUDA)>* ctx);

  virtual bool create(const operators::ConvParam& param,
                      Context<TARGET(kCUDA)>* ctx);

  virtual bool run(const operators::ConvParam& param);
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
