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

void TensorLite::ShareDataWith(const TensorLite &other) {
  buffer_ = other.buffer_;
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  precision_ = other.precision_;
  offset_ = other.offset_;
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
#ifdef LITE_WITH_XPU
  if (target_ != target && target == TargetType::kXPU) {
    buffer_.reset(new XPUBuffer);
  }
#endif
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
