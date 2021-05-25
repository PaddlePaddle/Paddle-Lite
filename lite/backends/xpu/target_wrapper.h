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

#include <memory>
#include <mutex>  // NOLINT
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/macros.h"

#define XPU_CALL(func)                                        \
  {                                                           \
    auto e = (func);                                          \
    CHECK_EQ(e, 0) << "XPU: (" << #func << ") returns " << e; \
  }

namespace paddle {
namespace lite {

// MAX(lod.size()) = 32
const int XPU_MAX_LOD_SIZE = 32;
// MAX(lod[i + 1] - lod[i]) = 512
const int XPU_MAX_LOD_SEQ_LEN = 512;

using TargetWrapperXPU = TargetWrapper<TARGET(kXPU)>;

struct XPUScratchPad {
  XPUScratchPad(void* addr, size_t size) : addr_(addr), size_(size) {}

  // XXX(miaotianxiang): |size_| increases monotonically
  void Reserve(size_t new_size);

  void* addr_{nullptr};
  size_t size_{0};
};

struct XPUScratchPadDeleter {
  void operator()(XPUScratchPad* sp) const;
};

using XPUScratchPadGuard = std::unique_ptr<XPUScratchPad, XPUScratchPadDeleter>;

template <>
class TargetWrapper<TARGET(kXPU)> {
 public:
  static size_t num_devices() { return 1; }
  static size_t maximum_stream() { return 0; }

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);

  static XPUScratchPadGuard MallocScratchPad(size_t size);

  static void MallocL3Cache(xdnn::Context* tls_raw_ctx,
                            size_t l3_size,
                            bool locked);
  static void FreeL3Cache(xdnn::Context* tls_raw_ctx,
                          size_t l3_size,
                          bool locked);
  static bool IsSharedL3Created() {
    return shared_l3_ptr_ == nullptr ? false : true;
  }
  // **DEPRECATED**, use xpu_set_device() at the very beginning of each worker
  // thread
  static void SetDev(int dev_no = 0) {
    const char* dev_env = getenv("LITE_XPU_DEV");
    if (dev_env) {
      dev_no = atoi(dev_env);
    }

    XPU_CALL(xpu_set_device(dev_no));
  }

  static size_t shared_l3_size;

 private:
  static void* shared_l3_ptr_;
  static std::mutex mutex_l3_;
};

}  // namespace lite
}  // namespace paddle
