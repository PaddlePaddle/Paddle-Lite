// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/api/paddle_place.h"
#include "lite/core/memory.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/log/logging.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

class XPUBuffer : public Buffer {
 public:
  XPUBuffer() {
    target_ = TargetType::kXPU;
    xpu_l3_cache_block_ = lite::TargetWrapperXPU::CreateL3CacheBlock();
  }

  void ResetLazy(TargetType target, size_t size) override {
    CHECK(target == TargetType::kXPU);
    CHECK_EQ(own_data_, true) << "Can not reset unowned buffer.";
    if (global_mem_space_ < size) {
      if (global_mem_space_ > 0) {
        lite::TargetWrapperXPU::Free(global_mem_data_);
      }
      global_mem_data_ = lite::TargetWrapperXPU::Malloc(size);
      global_mem_space_ = size;
    }
    if (xpu_l3_cache_block_ != nullptr) {
      xpu_l3_cache_block_->record(size);
      if (size <= xpu_l3_cache_block_->size()) {
        data_ = xpu_l3_cache_block_->data();
        space_ = xpu_l3_cache_block_->size();
      } else {
        data_ = global_mem_data_;
        space_ = global_mem_space_;
      }
    }
  }

  void CopyDataFrom(const Buffer& other, size_t nbytes) override {
    LOG(FATAL) << "Unsupport XPU D2D Memcpy";
  }

  void Free() {
    if (global_mem_space_ > 0) {
      lite::TargetWrapperXPU::Free(global_mem_data_);
    }
    global_mem_data_ = nullptr;
    global_mem_space_ = 0;
    data_ = nullptr;
    space_ = 0;
    target_ = TargetType::kHost;
  }

  ~XPUBuffer() { this->Free(); }

 private:
  XPUL3CacheBlock* xpu_l3_cache_block_{nullptr};
  size_t global_mem_space_{0};
  void* global_mem_data_{nullptr};
};

}  // namespace lite
}  // namespace paddle
