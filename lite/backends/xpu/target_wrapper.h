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

#include "lite/backends/xpu/runtime_option.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/xpu_l3_cache_block.h"
#include "lite/backends/xpu/xpu_l3_strategy.h"
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
// MAX(lod.size()) = 256 in XPU
const int XPU_MAX_LOD_SIZE_256 = 256;
// MAX(lod[i + 1] - lod[i]) = 512
const int XPU_MAX_LOD_SEQ_LEN = 512;

using TargetWrapperXPU = TargetWrapper<TARGET(kXPU)>;

template <>
class TargetWrapper<TARGET(kXPU)> {
 public:
  static size_t num_devices() { return 1; }
  static size_t maximum_stream() { return 0; }
  static void* get_xpu_stream() {
    // if ~LoadPredictorConfig is called before ~Predictor,
    // the xpu_runtime_ptr would be nullptr, so return nullptr make ~Predictor
    // go well
    if (xpu_runtime_ptr == nullptr) {
      return nullptr;
    }
    return xpu_runtime_ptr->xpu_stream.GetXPUStream();
  }

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
    if (xpu_runtime_ptr->xpu_tls_raw_ctx != nullptr) {
      xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext()->xpu_stream =
          xpu_runtime_ptr->xpu_stream.GetXPUStream();
      return xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext();
    }

    xpu_runtime_ptr->xpu_tls_raw_ctx.reset(new XDNNContext());
    xpu_runtime_ptr->xpu_tls_raw_ctx->CreatXDNNContext(
        xpu_runtime_ptr->xpu_dev_num);
    CHECK(xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext());
    xdnn::Context* xdnn_context =
        xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext();
    if (xpu_runtime_ptr->xpu_cluster_num != 0) {
      xdnn_context->set_ncluster(xpu_runtime_ptr->xpu_cluster_num);
    }

    if (xpu_runtime_ptr->xpu_sdnn_num != 0) {
      xdnn_context->set_nsdnn(xpu_runtime_ptr->xpu_sdnn_num);
    }
    if (xpu_runtime_ptr->xpu_enable_multi_stream) {
      // create different stream per thread
      xpu_runtime_ptr->xpu_stream.CreatXPUStream();
      CHECK(xpu_runtime_ptr->xpu_stream.GetXPUStream());
    }
    xdnn_context->xpu_stream = xpu_runtime_ptr->xpu_stream.GetXPUStream();
    xpu_runtime_ptr->xpu_stream.GetXPUStreamInfo();

    if (xdnn_context->dev().type() == xdnn::kXPU1) {
      LOG(INFO) << "running in KunLun1";
    } else if (xdnn_context->dev().type() == xdnn::kXPU2) {
      LOG(INFO) << "running in KunLun2";
    } else if (xdnn_context->dev().type() == xdnn::kXPU3) {
      LOG(INFO) << "running in KunLun3";
    } else {
      LOG(FATAL) << "running in unknown XPU device: "
                 << static_cast<int>(xdnn_context->dev().type());
    }
    if (xpu_runtime_ptr->xpu_fc_autotune_level > 0) {
      xdnn_context->set_option(
          "XPU_FC_AUTOTUNE",
          std::to_string(xpu_runtime_ptr->xpu_fc_autotune_level).c_str());
    }
    if (!xpu_runtime_ptr->xpu_fc_autotune_file.empty()) {
      xdnn_context->set_option("XPU_FC_AUTOTUNE_FILE",
                               xpu_runtime_ptr->xpu_fc_autotune_file.c_str());
    }
    if (xpu_runtime_ptr->xpu_fc_autotune_writeback) {
      xdnn_context->set_option(
          "XPU_FC_AUTOTUNE_WRITEBACK",
          std::to_string(xpu_runtime_ptr->xpu_fc_autotune_writeback).c_str());
    }
    if (xpu_runtime_ptr->quant_post_dynamic_activation_method > 0) {
      xdnn_context->set_option(
          "XPU_PRECISION_MODE",
          std::to_string(xpu_runtime_ptr->quant_post_dynamic_activation_method)
              .c_str());
    }
    if (xpu_runtime_ptr->transformer_softmax_optimize_level > 0) {
      xdnn_context->set_option(
          "XPU_SOFTMAX_OPT",
          std::to_string(xpu_runtime_ptr->transformer_softmax_optimize_level)
              .c_str());
    }
    if (xpu_runtime_ptr->xpu_l3_planner == nullptr) {
      xpu_runtime_ptr->xpu_l3_planner = new XPUL3Planner;
    }
    CHECK(xpu_runtime_ptr->xpu_l3_planner);
    xpu_runtime_ptr->xpu_l3_planner->set_l3_tune_level(
        GetIntFromEnv("L3_TUNE_LEVEL", 1));

    int devid = -1;
    uint64_t max_l3_size = 0;
    XPU_CALL(xpu_current_device(&devid));
    XPU_CALL(xpu_device_get_attr(
        &max_l3_size, XPUDeviceAttr(XPUATTR_MEM_L3_CAPACITY), devid));
    if (xpu_runtime_ptr->xpu_local_l3_size > max_l3_size) {
      xpu_runtime_ptr->xpu_local_l3_size = max_l3_size;
    }
    CHECK_LE(shared_l3_size, max_l3_size);
    if (xpu_runtime_ptr->xpu_local_gm_size > 0) {
      VLOG(3) << "Try To Malloc Local GM Workspace Size is "
              << xpu_runtime_ptr->xpu_local_gm_size;
      void* local_gm_ptr = nullptr;
      int ret = xpu_malloc(reinterpret_cast<void**>(&local_gm_ptr),
                           (xpu_runtime_ptr->xpu_local_gm_size));
      if (ret != 0 || local_gm_ptr == nullptr) {
        VLOG(3) << "No Enough GM Workspace For Current Predictor.";
      } else {
        ret = xdnn_context->_gm_mgr.set(local_gm_ptr,
                                        xpu_runtime_ptr->xpu_local_gm_size);
        if (ret != 0) {
          LOG(WARNING) << "XPU GM Mgr Init Fail, Please Check Configuration.";
          TargetWrapperXPU::Free(local_gm_ptr);
          local_gm_ptr = nullptr;
        }
      }
    }

    return xdnn_context;
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

  // TODO(quwei): refactor share l3.
  static LITE_THREAD_LOCAL bool need_l3_mutex;  // model level l3 size
  static size_t shared_l3_size;                 // model level l3 size
  static LITE_THREAD_LOCAL XPURunTimeOption* xpu_runtime_ptr;

 private:
  static void ScatterL3Cache(
      void* l3_ptr,
      size_t l3_size,
      const std::vector<std::vector<int64_t>>& query_shape);
  static void* shared_l3_ptr_;
  static std::mutex mutex_l3_;
};

class XPULoadRunTimeOptionGuard {
 public:
  XPULoadRunTimeOptionGuard(const XPULoadRunTimeOptionGuard&) = delete;
  XPULoadRunTimeOptionGuard& operator=(const XPULoadRunTimeOptionGuard&) =
      delete;
  explicit XPULoadRunTimeOptionGuard(XPURunTimeOption* xpu_runtime_option) {
    if (TargetWrapperXPU::xpu_runtime_ptr != xpu_runtime_option) {
      TargetWrapperXPU::xpu_runtime_ptr = xpu_runtime_option;
      // As rumtime context is thread_local,so we should set device when
      // using different predictor in the same thread.
      XPU_CALL(xpu_set_device(TargetWrapperXPU::xpu_runtime_ptr->xpu_dev_num));
    }
  }
  ~XPULoadRunTimeOptionGuard() { TargetWrapperXPU::xpu_runtime_ptr = nullptr; }
};

}  // namespace lite
}  // namespace paddle
