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
#include <string>
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

  void CreatXDNNContext(int devid) {
    int cur_dev_id = -1;
    XPU_CALL(xpu_current_device(&cur_dev_id));
    CHECK_EQ(cur_dev_id, devid)
        << "XPU context config device id is :" << devid
        << ",but we get current device id is : " << cur_dev_id;
    rawcontext_ = xdnn::create_context();
    devid_ = devid;
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
    if (xpu_stream_origin_ != nullptr) {
      XPU_CALL(xpu_set_device(devid_));
      VLOG(6) << "thread 0x" << std::hex << std::this_thread::get_id()
              << " Destory context xpu stream: " << xpu_stream_origin_;
      CHECK(xpu_stream_destroy(xpu_stream_origin_) == 0)
          << "xpu stream destroy failed.";
      xpu_stream_origin_ = nullptr;
    }
  }

  XPUStream(const XPUStream&) = delete;
  XPUStream& operator=(const XPUStream&) = delete;

  // TODO(quwei): Only creat one non-default xpu stream in current case,but
  // maybe use more non-default xpu stream in the future.
  void CreatXPUStream() {
    if (xpu_stream_ != nullptr) {
      VLOG(3) << " xpu stream not null before create,so we will not creat new "
                 "stream.Current stream addr is : "
              << xpu_stream_;
      return;
    }

    int cur_dev_id = -1;
    XPU_CALL(xpu_current_device(&cur_dev_id));
    CHECK_EQ(cur_dev_id, devid_)
        << "XPU Stream config device id is :" << devid_
        << ",but we get current device id is : " << cur_dev_id;

    CHECK(xpu_stream_create(&xpu_stream_) == 0) << "xpu stream_create failed.";
    LOG(INFO) << "thread 0x" << std::hex << std::this_thread::get_id()
              << " Creat context xpu stream: " << xpu_stream_;
    CHECK(xpu_stream_ != nullptr);
    xpu_stream_origin_ = xpu_stream_;
  }

  void* GetXPUStream() { return xpu_stream_; }
  void SetXPUDevid(int devid) { devid_ = devid; }
  void SetXPUStream(void* xpu_stream) {
    CHECK(xpu_stream != nullptr) << "Not set default stream.";
    // TODO(quwei): Check  devid in current stream.
    if (xpu_stream_origin_ != nullptr) {
      CHECK(xpu_stream_destroy(xpu_stream_origin_) == 0)
          << "xpu stream destroy failed.";
      LOG(WARNING) << "Thread 0x" << std::hex << std::this_thread::get_id()
                   << ",Destory origin xpu stream: " << xpu_stream_origin_
                   << ",which create by lite context.";
      xpu_stream_origin_ = nullptr;
    }

    xpu_stream_ = xpu_stream;
  }

 private:
  void* xpu_stream_{nullptr};
  // used to record origin stream which create by lite.
  void* xpu_stream_origin_{nullptr};
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
    xpu_stream.SetXPUDevid(xpu_dev_num);
    if (!config->multi_encoder_precision.empty()) {
      multi_encoder_precision = config->multi_encoder_precision;
    }
    if (!config->compute_precision.empty()) {
      compute_precision = config->compute_precision;
    }
    multi_encoder_adaptive_seqlen = config->multi_encoder_adaptive_seqlen;
    local_quant = config->local_quant;
    if (!config->xpu_dump_log_path.empty()) {
      xpu_dump_log_path = config->xpu_dump_log_path;
      need_dump_xpu_info = true;
    }
    if (!config->xpu_dump_tensor_path.empty()) {
      xpu_dump_tensor_path = config->xpu_dump_tensor_path;
      need_dump_xpu_info = true;
    }
    // Perdictor clone need set device.
    XPU_CALL(xpu_set_device(xpu_dev_num));
  }

  // set by config
  size_t xpu_local_l3_size{std::numeric_limits<size_t>::max()};
  bool xpu_local_l3_autotune{true};
  size_t xpu_local_gm_size{0};
  int xpu_cluster_num{0};
  int xpu_sdnn_num{0};
  bool xpu_enable_multi_stream{false};
  int xpu_dev_num{0};
  // encoder config
  std::string multi_encoder_precision;
  std::string compute_precision;
  bool multi_encoder_adaptive_seqlen{false};
  bool local_quant{false};
  // dump tensor
  std::string xpu_dump_tensor_path{""};
  std::string xpu_dump_log_path{""};
  bool need_dump_xpu_info{false};
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
