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

namespace paddle {
namespace lite {
namespace cuda {
namespace math {
namespace cudnn {

template <typename T>
class cudnnTypeWrapper;

template <>
class cudnnTypeWrapper<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  typedef const float ScalingParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0f;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0f;
    return &v;
  }
};

template <>
class cudnnTypeWrapper<half> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
  typedef const half ScalingParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = __float2half(1.0f);
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = __float2half(0.0f);
    return &v;
  }
};

struct ParamsRegion {
  ParamsRegion() : offset_(nullptr), size_(0) {}
  ParamsRegion(void* offset, size_t size) : offset_(offset), size_(size) {}
  ~ParamsRegion() {}

  ParamsRegion& operator=(const ParamsRegion& right) {
    offset_ = right.offset_;
    size_ = right.size_;
    return *this;
  }

  bool operator==(const ParamsRegion& right) {
    bool comp_eq = true;
    comp_eq = comp_eq && (offset_ == right.offset_);
    comp_eq = comp_eq && (size_ = right.size_);
    return comp_eq;
  }

  void* offset_;
  size_t size_;
};

template <typename T>
class TensorDescriptors {
 public:
  TensorDescriptors(size_t n,
                    const std::vector<std::vector<int>>& dim,
                    const std::vector<std::vector<int>>& stride) {
    descs_.resize(n);
    CHECK_EQ(dim.size(), stride.size())
        << "dim size should be equal to stride size";
    for (size_t i = 0; i < n; ++i) {
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs_[i]));
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(descs_[i],
                                             cudnnTypeWrapper<T>::type,
                                             dim[i].size(),
                                             dim[i].data(),
                                             stride[i].data()));
    }
  }

  ~TensorDescriptors() {
    for (auto desc : descs_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    }
  }

  const cudnnTensorDescriptor_t* descs() const { return descs_.data(); }

  int size() const { return descs_.size(); }

 private:
  std::vector<cudnnTensorDescriptor_t> descs_;
};

}  // namespace cudnn
}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
