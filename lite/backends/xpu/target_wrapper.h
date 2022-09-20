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
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/xpu_l3_cache_block.h"
#include "lite/backends/xpu/xpu_l3_strategy.h"
#include "lite/backends/xpu/xpu_quantizer.h"
#include "lite/backends/xpu/xpu_scratch.h"
#include "lite/core/dim.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/env.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

// MAX(lod.size()) = 32
const int XPU_MAX_LOD_SIZE = 32;
// MAX(lod.size()) = 64 in XPU refactor
const int XPU_MAX_LOD_SIZE_64 = 64;
// MAX(lod[i + 1] - lod[i]) = 512
const int XPU_MAX_LOD_SEQ_LEN = 512;

using TargetWrapperXPU = TargetWrapper<TARGET(kXPU)>;

template <>
class TargetWrapper<TARGET(kXPU)> {
 public:
  static size_t num_devices() { return 1; }
  static size_t maximum_stream() { return 0; }
  static void enable_xpu_multi_stream() { enable_multi_stream_ = true; }
  static bool xpu_multi_stream() { return enable_multi_stream_; }
  static void* get_xpu_stream() { return xpu_stream_.get(); }

  static void* Malloc(size_t size) { return XPUMemory::Malloc(size); }
  static void Free(void* ptr) { XPUMemory::Free(ptr); }

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);

  static XPUScratchPadGuard MallocScratchPad(size_t size) {
    return XPUMemory::MallocScratchPad(size);
  }

  template <typename Tcpu, typename Txpu>
  static XPUQuantData ConvertCPUWeightToXPUQuantWeight(const Tcpu* cpu_data,
                                                       const DDimLite& dims,
                                                       bool data_transpose,
                                                       size_t max_ptr_len);

  static xdnn::Context* GetRawContext() {
    if (tls_raw_ctx_.get() == nullptr) {
      tls_raw_ctx_.reset(xdnn::create_context(), xdnn::destroy_context);
      CHECK(tls_raw_ctx_.get());
      if (cluster_num != 0) {
        tls_raw_ctx_->set_ncluster(cluster_num);
      }
      if (sdnn_num != 0) {
        tls_raw_ctx_->set_nsdnn(sdnn_num);
      }
      if (!enable_multi_stream_) {
        CHECK(xpu_stream_.get() == nullptr)
            << " xpu default stream should be nullptr: " << xpu_stream_.get();
        VLOG(3) << "all threads share the default xpu stream";
      } else {
        // use different stream per thread
        CHECK(xpu_stream_.get() == nullptr)
            << " xpu stream not null before create: " << xpu_stream_.get();
        void* tls_xpu_stream = nullptr;
        CHECK(xpu_stream_create(&tls_xpu_stream) == 0)
            << "xpu_stream_create failed";
        CHECK(tls_xpu_stream != nullptr);
        xpu_stream_.reset(tls_xpu_stream, xpu_stream_destroy);
        CHECK(xpu_stream_.get());
      }
      tls_raw_ctx_.get()->xpu_stream = xpu_stream_.get();
      if (tls_raw_ctx_.get()->dev().type() == xdnn::kXPU1) {
        LOG(INFO) << "running in KunLun1";
      } else if (tls_raw_ctx_.get()->dev().type() == xdnn::kXPU2) {
        LOG(INFO) << "running in KunLun2";
      } else if (tls_raw_ctx_.get()->dev().type() == xdnn::kXPU3) {
        LOG(INFO) << "running in KunLun3";
      } else {
        LOG(FATAL) << "running in unknown XPU device: "
                   << static_cast<int>(tls_raw_ctx_.get()->dev().type());
      }
      LOG(INFO) << "thread 0x" << std::hex << std::this_thread::get_id()
                << " set context xpu stream: " << xpu_stream_.get();
      if (l3_planner_ == nullptr) {
        l3_planner_ = new XPUL3Planner;
      }
      CHECK(l3_planner_);
      if (quantizer_.get() == nullptr) {
        quantizer_.reset(new XPUQuantizer());
      }
      CHECK(quantizer_.get());
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
        VLOG(3) << "Try To Malloc Local GM Workspace Size is " << local_gm_size;
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
    return tls_raw_ctx_.get();
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
  static LITE_THREAD_LOCAL std::string compute_precision;  // NOLINT
  // only for R200
  static LITE_THREAD_LOCAL bool local_quant;
  // l3 cache config
  static LITE_THREAD_LOCAL bool need_l3_mutex;    // model level l3 size
  static LITE_THREAD_LOCAL size_t local_l3_size;  // model level l3 size
  static LITE_THREAD_LOCAL bool local_l3_autotune;
  static LITE_THREAD_LOCAL size_t local_gm_size;
  static size_t shared_l3_size;  // model level l3 size
  static LITE_THREAD_LOCAL std::vector<XPUL3CacheBlock*>
      l3_block_dict;  // l3 cache block used between op layers
  static LITE_THREAD_LOCAL int cluster_num;
  static LITE_THREAD_LOCAL int sdnn_num;

 private:
  static void ScatterL3Cache(
      void* l3_ptr,
      size_t l3_size,
      const std::vector<std::vector<int64_t>>& query_shape);
  static LITE_THREAD_LOCAL std::shared_ptr<xdnn::Context> tls_raw_ctx_;
  static LITE_THREAD_LOCAL std::shared_ptr<void> xpu_stream_;
  static LITE_THREAD_LOCAL void* local_l3_ptr_;
  static void* shared_l3_ptr_;
  static std::mutex mutex_l3_;
  static bool enable_multi_stream_;
  static LITE_THREAD_LOCAL XPUL3Planner* l3_planner_;
  static LITE_THREAD_LOCAL std::shared_ptr<XPUQuantizer> quantizer_;
};

}  // namespace lite
}  // namespace paddle
