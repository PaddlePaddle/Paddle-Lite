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

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/xpu_l3_cache_block.h"
#include "lite/backends/xpu/xpu_l3_strategy.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/log/cp_logging.h"
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
// QUANT SCALE NUM == XPU CDNN NUM
const int XPU_QUANT_SCALE_NUM = 6;

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

  static xdnn::Context* GetRawContext() {
    if (tls_raw_ctx_ == nullptr) {
      tls_raw_ctx_ = xdnn::create_context();
      CHECK(tls_raw_ctx_);
      if (l3_planner_ == nullptr) {
        l3_planner_ = new XPUL3Planner;
      }
      CHECK(l3_planner_);
      if (conv_autotune) {
        tls_raw_ctx_->_xpu1_conv_selector.set_autotune_loop(true);
        tls_raw_ctx_->_xpu1_conv_selector.set_inference_mode(true);
      }
      if (!conv_autotune_file.empty()) {
        tls_raw_ctx_->_xpu1_conv_selector.set_autotune_file(
            conv_autotune_file.c_str());
      }
      int devid = -1;
      uint64_t max_l3_size = 0;
      XPU_CALL(xpu_current_device(&devid));
      XPU_CALL(xpu_device_get_attr(
          &max_l3_size, XPUDeviceAttr(XPUATTR_MEM_L3_CAPACITY), devid));
      if (local_l3_size > max_l3_size) {
        local_l3_size = max_l3_size;
      }
      CHECK_LE(shared_l3_size, max_l3_size);
      if (local_gm_size > 0) {
        VLOG(3) << "Try To Malloc Local GM Workspace Size is" << local_gm_size;
        void* local_gm_ptr = nullptr;
        int ret =
            xpu_malloc(reinterpret_cast<void**>(&local_gm_ptr), local_gm_size);
        if (ret != 0 || local_gm_ptr == nullptr) {
          VLOG(3) << "No Enough GM Workspace For Current Predictor.";
        } else {
          void* old_ptr = tls_raw_ctx_->_gm_mgr.get_ptr();
          if (old_ptr != nullptr) {
            TargetWrapperXPU::Free(old_ptr);
          }
          ret = tls_raw_ctx_->_gm_mgr.set(local_gm_ptr, local_gm_size);
          if (ret != 0) {
            LOG(WARNING) << "XPU GM Mgr Init Fail, Please Check Configuration.";
            TargetWrapperXPU::Free(local_gm_ptr);
            local_gm_ptr = nullptr;
          }
        }
      }
    }
    return tls_raw_ctx_;
  }
  static void MallocL3Cache(
      const std::vector<std::vector<int64_t>>& query_shape);
  static void FreeL3Cache();
  static bool IsSharedL3Created() {
    return shared_l3_ptr_ == nullptr ? false : true;
  }
  static XPUL3CacheBlock* CreateL3CacheBlock();

  // **DEPRECATED**, use xpu_set_device() at the very beginning of each worker
  // thread
  static void SetDev(int dev_no = 0) {
    const char* dev_env = getenv("LITE_XPU_DEV");
    if (dev_env) {
      dev_no = atoi(dev_env);
    }

    XPU_CALL(xpu_set_device(dev_no));
  }

  // multi encoder config
  static LITE_THREAD_LOCAL std::string multi_encoder_precision;  // NOLINT
  static LITE_THREAD_LOCAL bool multi_encoder_adaptive_seqlen;
  // conv autotune config
  static LITE_THREAD_LOCAL bool conv_autotune;
  static LITE_THREAD_LOCAL std::string conv_autotune_file;  // NOLINT
  // l3 cache config
  static LITE_THREAD_LOCAL bool need_l3_mutex;    // model level l3 size
  static LITE_THREAD_LOCAL size_t local_l3_size;  // model level l3 size
  static LITE_THREAD_LOCAL size_t local_gm_size;
  static size_t shared_l3_size;  // model level l3 size
  static LITE_THREAD_LOCAL std::vector<XPUL3CacheBlock*>
      l3_block_dict;  // l3 cache block used between op layers

 private:
  static void ScatterL3Cache(
      void* l3_ptr,
      size_t l3_size,
      const std::vector<std::vector<int64_t>>& query_shape);
  static LITE_THREAD_LOCAL xdnn::Context* tls_raw_ctx_;
  static LITE_THREAD_LOCAL void* local_l3_ptr_;
  static void* shared_l3_ptr_;
  static std::mutex mutex_l3_;
  static LITE_THREAD_LOCAL XPUL3Planner* l3_planner_;
};

}  // namespace lite
}  // namespace paddle
