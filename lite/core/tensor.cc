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

#ifndef LITE_WITH_FPGA

#include "lite/core/tensor.h"
#include <string>
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

using value_type = int64_t;

value_type DDimLite::production() const {
  value_type res = 1;
  for (size_t i = 0; i < this->size(); i++) {
    res *= (*this)[i];
  }
  return res;
}

value_type DDimLite::count(int start, int end) const {
  if (start < 0) {
    start = 0;
  }
  if (end > size()) {
    end = size();
  }
  if (end < start) {
    end = start;
  }
  value_type sum = 1;
  for (auto i = start; i < end; ++i) {
    sum *= data_[i];
  }
  return sum;
}

DDimLite DDimLite::Slice(int start, int end) const {
  std::vector<value_type> vec;
  for (int i = start; i < end; i++) {
    vec.push_back((*this)[i]);
  }
  return DDimLite(vec);
}

std::string DDimLite::repr() const {
  STL::stringstream ss;
  if (empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < this->size() - 1; i++) {
    ss << (*this)[i] << ",";
  }
  if (!this->empty()) ss << (*this)[size() - 1];
  ss << "}";
  return ss.str();
}

void TensorLite::ShareDataWith(const TensorLite &other) {
  buffer_ = other.buffer_;
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
}

void TensorLite::CopyDataFrom(const TensorLite &other) {
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  buffer_->CopyDataFrom(*other.buffer_, memory_size_);
}

void *TensorLite::mutable_data(size_t memory_size) {
  memory_size_ = memory_size;
  buffer_->ResetLazy(target_, memory_size_);
  return buffer_->data();
}

void *TensorLite::mutable_data(TargetType target, size_t memory_size) {
  target_ = target;
  return mutable_data(memory_size);
}

#ifdef LITE_WITH_OPENCL
// full specialization of TensorLite::mutable_data() for opencl Image2D
template <>
cl::Image2D *TensorLite::mutable_data<int8_t, cl::Image2D>() {
  memory_size_ = dims_.production() * sizeof(int8_t);
  std::array<size_t, 2> image2d_shape{1, 1};
  std::array<size_t, 2> image2d_pitch{1, 1};
  convertDimsToImage2DShape(image2d_shape, image2d_pitch);
  buffer_->ResetLazyImage2D<int8_t>(target_, image2d_shape, image2d_pitch);
  return reinterpret_cast<cl::Image2D *>(static_cast<char *>(buffer_->data()) +
                                         offset_);
}

template <>
cl::Image2D *TensorLite::mutable_data<int32_t, cl::Image2D>() {
  memory_size_ = dims_.production() * sizeof(int32_t);
  std::array<size_t, 2> image2d_shape{1, 1};
  std::array<size_t, 2> image2d_pitch{1, 1};
  convertDimsToImage2DShape(image2d_shape, image2d_pitch);
  buffer_->ResetLazyImage2D<int8_t>(target_, image2d_shape, image2d_pitch);
  return reinterpret_cast<cl::Image2D *>(static_cast<char *>(buffer_->data()) +
                                         offset_);
}

template <>
cl::Image2D *TensorLite::mutable_data<float, cl::Image2D>() {
  memory_size_ = dims_.production() * sizeof(float);
  std::array<size_t, 2> image2d_shape{1, 1};
  std::array<size_t, 2> image2d_pitch{1, 1};
  convertDimsToImage2DShape(image2d_shape, image2d_pitch);
  buffer_->ResetLazyImage2D<int8_t>(target_, image2d_shape, image2d_pitch);
  return reinterpret_cast<cl::Image2D *>(static_cast<char *>(buffer_->data()) +
                                         offset_);
}

// full specialization of TensorLite::mutable_data(TargetType target) for opencl
// Image2D
template <>
cl::Image2D *TensorLite::mutable_data<int8_t, cl::Image2D>(TargetType target) {
  target_ = target;
  return TensorLite::mutable_data<int8_t, cl::Image2D>();
}

template <>
cl::Image2D *TensorLite::mutable_data<int32_t, cl::Image2D>(TargetType target) {
  target_ = target;
  return TensorLite::mutable_data<int32_t, cl::Image2D>();
}

template <>
cl::Image2D *TensorLite::mutable_data<float, cl::Image2D>(TargetType target) {
  target_ = target;
  return TensorLite::mutable_data<float, cl::Image2D>();
}
#endif  // LITE_WITH_OPENCL

// static LoD TensorLite::ToAbsOffset(const LoD &lod) {
//  if (lod.empty() || lod.size() == 1) return lod;
//  LoD ret = lod;
//  for (int level = static_cast<int>(lod.size()) - 2; level >= 0; --level) {
//    for (size_t i = 0; i < lod[level].size(); ++i) {
//      size_t index = lod[level][i];
//      result[level][i] = result[level + 1][index];
//    }
//  }
//}

}  // namespace lite
}  // namespace paddle

#endif  // #ifndef LITE_WITH_FPGA
