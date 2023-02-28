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

#include "runtime/device.h"
#include <stdlib.h>
#include "utilities/logging.h"

namespace adnn {
namespace runtime {

void* OpenDevice(int thread_num) { return nullptr; }

void CloseDevice(void* device) {}

void* CreateContext(void* device, int thread_num) { return nullptr; }

void DestroyContext(void* context) {}

void* Alloc(void* context, size_t size) { return malloc(size); }

void Free(void* context, void* ptr) {
  if (ptr) {
    free(ptr);
  }
}

void* AlignedAlloc(void* context, size_t alignment, size_t size) {
  size_t offset = sizeof(void*) + alignment - 1;
  char* p = static_cast<char*>(malloc(offset + size));
  // Byte alignment
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(alignment - 1)));
  static_cast<void**>(r)[-1] = p;
  return r;
}

void AlignedFree(void* context, void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

}  // namespace runtime
}  // namespace adnn

adnn::Callback g_DefaultCallback = {
    .open_device = adnn::runtime::OpenDevice,
    .close_device = adnn::runtime::CloseDevice,
    .create_context = adnn::runtime::CreateContext,
    .destroy_context = adnn::runtime::DestroyContext,
    .alloc = adnn::runtime::Alloc,
    .free = adnn::runtime::Free,
    .aligned_alloc = adnn::runtime::AlignedAlloc,
    .aligned_free = adnn::runtime::AlignedFree,
};

namespace adnn {
namespace runtime {

Device::Device(int thread_num, const Callback* callback)
    : thread_num_(thread_num), callback_(callback) {
  if (!callback_) {
    callback_ = &g_DefaultCallback;
  }
  ADNN_CHECK(callback_->open_device);
  device_ = callback_->open_device(thread_num_);
}

void* Device::CreateContext(int thread_num) {
  if (callback_) {
    ADNN_CHECK(callback_->create_context);
    return callback_->create_context(device_, thread_num);
  }
  return nullptr;
}

void Device::DestroyContext(void* context) {
  if (callback_) {
    ADNN_CHECK(callback_->destroy_context);
    callback_->destroy_context(context);
  }
}

void* Device::Alloc(void* context, size_t size) {
  if (callback_) {
    ADNN_CHECK(callback_->alloc);
    return callback_->alloc(context, size);
  }
  return nullptr;
}

void Device::Free(void* context, void* ptr) {
  if (callback_) {
    ADNN_CHECK(callback_->free);
    callback_->free(context, ptr);
  }
}

void* Device::AlignedAlloc(void* context, size_t alignment, size_t size) {
  if (callback_) {
    ADNN_CHECK(callback_->aligned_alloc);
    return callback_->aligned_alloc(context, alignment, size);
  }
  return nullptr;
}

void Device::AlignedFree(void* context, void* ptr) {
  if (callback_) {
    ADNN_CHECK(callback_->aligned_free);
    callback_->aligned_free(context, ptr);
  }
}

Device::~Device() {
  if (callback_) {
    ADNN_CHECK(callback_->close_device);
    callback_->close_device(device_);
  }
  callback_ = nullptr;
  device_ = nullptr;
}

}  // namespace runtime
}  // namespace adnn
