// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <limits>
#include <memory>
#include <mutex>  //NOLINT
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/xpu_l3_cache_block.h"
#include "lite/backends/xpu/xpu_l3_strategy.h"
#include "lite/backends/xpu/xpu_quantizer.h"
#include "lite/backends/xpu/xpu_scratch.h"

namespace paddle {
namespace lite {

class XDNNContext {
 public:
  XDNNContext() {}
  ~XDNNContext() {
    XPU_CALL(xpu_set_device(devid_));
    xdnn::destroy_context(rawcontext_);
    VLOG(6) << "Destroy xpu context.";
  }

  XDNNContext(const XDNNContext&) = delete;
  XDNNContext& operator=(const XDNNContext&) = delete;

  void CreatXDNNContext() {
    XPU_CALL(xpu_current_device(&devid_));
    rawcontext_ = xdnn::create_context();
  }

  xdnn::Context* GetXDNNContext() { return rawcontext_; }

 private:
  xdnn::Context* rawcontext_{nullptr};
  int devid_{-1};
};

class XPUStream {
 public:
  XPUStream() {}
  ~XPUStream() {
    if (xpu_stream_ != nullptr) {
      XPU_CALL(xpu_set_device(devid_));
      VLOG(6) << "thread 0x" << std::hex << std::this_thread::get_id()
              << " Destory context xpu stream: " << xpu_stream_;
      CHECK(xpu_stream_destroy(xpu_stream_) == 0)
          << "xpu stream destroy failed.";
    }
  }

  XPUStream(const XPUStream&) = delete;
  XPUStream& operator=(const XPUStream&) = delete;

  // TODO(quwei): Only creat one non-default xpu stream in current case,but
  // maybe use more non-default xpu stream in the future.
  void CreatXPUStream() {
    XPU_CALL(xpu_current_device(&devid_));
    CHECK(xpu_stream_create(&xpu_stream_) == 0) << "xpu stream_create failed.";
    VLOG(6) << "thread 0x" << std::hex << std::this_thread::get_id()
            << " Creat context xpu stream: " << xpu_stream_;
    CHECK(xpu_stream_ != nullptr);
  }

  void* GetXPUStream() { return xpu_stream_; }

 private:
  void* xpu_stream_{nullptr};
  int devid_{-1};
};

struct XPURunTimeOption {
  void Set(const XPURunTimeOption* config) {
    xpu_local_l3_size = config->xpu_local_l3_size;
    xpu_local_l3_autotune = config->xpu_local_l3_autotune;
    xpu_local_gm_size = config->xpu_local_gm_size;
    xpu_cluster_num = config->xpu_cluster_num;
    xpu_sdnn_num = config->xpu_sdnn_num;
    xpu_enable_multi_stream = config->xpu_enable_multi_stream;
    xpu_dev_num = config->xpu_dev_num;
  }

  // set by config
  size_t xpu_local_l3_size{std::numeric_limits<size_t>::max()};
  bool xpu_local_l3_autotune{true};
  size_t xpu_local_gm_size{0};
  int xpu_cluster_num{0};
  int xpu_sdnn_num{0};
  bool xpu_enable_multi_stream{false};
  int xpu_dev_num{0};

  // Set in runtime
  std::vector<XPUL3CacheBlock*>
      xpu_l3_block_dict;  // l3 cache block used between op layers
  void* xpu_local_l3_ptr{nullptr};
  XPUL3Planner* xpu_l3_planner{nullptr};
  XPUStream xpu_stream;
  std::unique_ptr<XDNNContext> xpu_tls_raw_ctx{nullptr};
  XPUQuantizer quantizer;
};

}  // namespace lite
}  // namespace paddle
