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
  for (size_t i = 0; i < data_.size(); i++) {
    res *= data_[i];
  }
  return res;
}

value_type DDimLite::count(int start, int end) const {
  start = (std::max)(start, 0);
  end = (std::min)(end, static_cast<int>(data_.size()));
  if (end < start) {
    return 0;
  }
  value_type sum = 1;
  for (auto i = start; i < end; ++i) {
    sum *= data_[i];
  }
  return sum;
}

DDimLite DDimLite::Slice(int start, int end) const {
  start = (std::max)(start, 0);
  end = (std::min)(end, static_cast<int>(data_.size()));
  std::vector<value_type> new_dim(end - start);
  for (int i = start; i < end; i++) {
    new_dim[i - start] = data_[i];
  }
  return DDim(new_dim);
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
  precision_ = other.precision_;
}

void TensorLite::CopyDataFrom(const TensorLite &other) {
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  precision_ = other.precision_;
  persistable_ = other.persistable_;
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

void TensorLite::ResetBuffer(std::shared_ptr<Buffer> buffer,
                             size_t memory_size) {
  CHECK_EQ(offset_, 0u)
      << "Only the offset is supported to zero when the Buffer is reset.";
  if (buffer_) {
    CHECK_LE(memory_size_, buffer->space())
        << "The space of buffer is not enough to store the tensor.";
    CHECK_LE(memory_size, buffer->space())
        << "The buffer is smaller than the specified minimum size.";
  }
  buffer_ = buffer;
  memory_size_ = memory_size;
  target_ = buffer->target();
}

#ifdef LITE_WITH_OPENCL
template <>
const cl::Image2D *TensorLite::data<float, cl::Image2D>() const {
  if (nullptr == buffer_->data()) return nullptr;
  return static_cast<const cl::Image2D *>(buffer_->data());
}

template <>  // use uint16_t represent half float
const cl::Image2D *TensorLite::data<uint16_t, cl::Image2D>() const {
  if (nullptr == buffer_->data()) return nullptr;
  return static_cast<const cl::Image2D *>(buffer_->data());
}
#endif

}  // namespace lite
}  // namespace paddle

#endif  // #ifndef LITE_WITH_FPGA
