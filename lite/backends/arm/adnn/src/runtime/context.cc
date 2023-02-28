// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "runtime/context.h"
#include "utilities/logging.h"

namespace adnn {
namespace runtime {

Context::Context(Device* device, int thread_num)
    : device_(device), thread_num_(thread_num) {
  ADNN_CHECK(device_);
  context_ = device_->CreateContext(thread_num_);
}

void* Context::Alloc(size_t size) {
  ADNN_CHECK(device_);
  return device_->Alloc(context_, size);
}

void Context::Free(void* ptr) {
  ADNN_CHECK(device_);
  return device_->Free(context_, ptr);
}

void* Context::AlignedAlloc(size_t alignment, size_t size) {
  ADNN_CHECK(device_);
  return device_->AlignedAlloc(context_, alignment, size);
}

void Context::AlignedFree(void* ptr) {
  ADNN_CHECK(device_);
  return device_->AlignedFree(context_, ptr);
}

Context::~Context() {
  if (device_ && context_) {
    device_->DestroyContext(context_);
  }
  device_ = nullptr;
  context_ = nullptr;
}

}  // namespace runtime
}  // namespace adnn
