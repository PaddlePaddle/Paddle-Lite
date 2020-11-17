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

#pragma once

#include <memory>
#include <utility>
#include "lite/core/memory.h"

namespace paddle {
namespace lite {
namespace fbs {

// A simple inline package of core::Buffer.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;

  explicit Buffer(size_t size)
      : raw_{std::unique_ptr<lite::Buffer>{new lite::Buffer}} {
    ReallocateDownward(size);
  }

  void CopyDataFrom(const Buffer& other) {
    raw_ = std::unique_ptr<lite::Buffer>{new lite::Buffer};
    const auto* other_raw = other.raw();
    CHECK(other_raw);
    raw_->CopyDataFrom(*other_raw, other.size());
  }

  Buffer(Buffer&& other) { raw_ = other.Release(); }
  Buffer& operator=(Buffer&& other) {
    raw_ = other.Release();
    return *this;
  }

  const void* data() const {
    CHECK(raw_);
    return raw_->data();
  }
  void* data() {
    CHECK(raw_);
    return raw_->data();
  }
  size_t size() const {
    CHECK(raw_);
    return raw_->space();
  }
  void ReallocateDownward(size_t size) {
    CHECK(raw_);
    return raw_->ResetLazy(TargetType::kHost, size);
  }

  std::unique_ptr<lite::Buffer> Release() { return std::move(raw_); }
  const lite::Buffer* raw() const { return raw_.get(); }

 private:
  std::unique_ptr<lite::Buffer> raw_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
