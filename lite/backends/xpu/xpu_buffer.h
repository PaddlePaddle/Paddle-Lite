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

namespace paddle {
namespace lite {
namespace xpu {

class XPUBuffer : public Buffer {
 public:
  void* data() const override { 
    if (target_ != TargetType::kXPU) {
      LOG(FATAL) << "Wrong Target: kXPU";
    }
    if (xpu_l3_cache_block_ != nullptr &&
        xpu_l3_cache_block_->in_use_ == true) {
      return xpu_l3_cache_block_->data();
    } else {
      return data_;
    }
  }
  
  void ResetLazy(TargetType target, size_t size) override {
    if (target_ != TargetType::kXPU) {
      LOG(FATAL) << "Wrong Target: kXPU";
    }
    if (target != target_ || space_ < size) {
      CHECK_EQ(own_data_, true) << "Can not reset unowned buffer.";
      
      target_ = target;
      space_ = size;
      if (xpu_l3_cache_block_ == nullptr) {
        Free();
        data_ = TargetMalloc(target, size);
      } else {
        xpu_l3_cache_block_->record(size);
        if (size <= xpu_l3_cache_block_->size()) {
          VLOG(4) << "TRUE, Acquire Size is " << size << ", L3 Size is "
                  << xpu_l3_cache_block_->size();
          xpu_l3_cache_block_->in_use_ = true;
        } else {
          VLOG(4) << "False, Acquire Size is " << size << ", L3 Size is "
                  << xpu_l3_cache_block_->size();
          xpu_l3_cache_block_->in_use_ = false;
          Free();
          data_ = TargetMalloc(target, size);
        }
      }
    }
  }

 private:
  XPUL3CacheBlock *xpu_l3_cache_block_{nullptr};

};

}  // namespace lite
}  // namespace paddle
}  // namespace xpu