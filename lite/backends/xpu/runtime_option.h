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
#include "lite/utils/env.h"

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

  void* GetXPUStream() const { return xpu_stream_; }
  void SetXPUDevid(int devid) { devid_ = devid; }
  void SetXPUStream(void* xpu_stream) {
    // TODO(quwei): Check  devid in current stream.
    if ((xpu_stream_origin_ != nullptr) && (xpu_stream_origin_ != xpu_stream)) {
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
  // config to predictor's XPURunTimeOption config
  void Set(const XPURunTimeOption* config) {
    xpu_local_l3_size = config->xpu_local_l3_size;
    xpu_local_l3_autotune = config->xpu_local_l3_autotune;
    xpu_local_gm_size = config->xpu_local_gm_size;
    xpu_cluster_num = config->xpu_cluster_num;
    xpu_sdnn_num = config->xpu_sdnn_num;
    xpu_enable_multi_stream = config->xpu_enable_multi_stream;
    xpu_dev_num = config->xpu_dev_num;
    xpu_stream.SetXPUDevid(xpu_dev_num);
    if (config->xpu_stream.GetXPUStream()) {
      xpu_stream.SetXPUStream(config->xpu_stream.GetXPUStream());
    }
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
    xpu_fc_autotune_level = config->xpu_fc_autotune_level;
    if (!config->xpu_fc_autotune_file.empty()) {
      xpu_fc_autotune_file = config->xpu_fc_autotune_file;
    }
    xpu_fc_autotune_writeback = config->xpu_fc_autotune_writeback;
    quant_post_dynamic_activation_method =
        config->quant_post_dynamic_activation_method;
    transformer_softmax_optimize_level =
        config->transformer_softmax_optimize_level;
    quant_post_static_gelu_out_threshold =
        config->quant_post_static_gelu_out_threshold;
    api_l3_reserve = static_cast<size_t>(
        GetIntFromEnv("API_L3_SIZE_RES", config->api_l3_reserve));
    CHECK_LE(api_l3_reserve, xpu_local_l3_size)
        << "api_l3_reserve should <= local_l3_size";
    // Perdictor clone need set device.
    XPU_CALL(xpu_set_device(xpu_dev_num));
  }

  // usually use XPU_VISIBLE_DEVICES, it's not neccessary
  int xpu_dev_num{0};
  // set by config
  size_t xpu_local_l3_size{std::numeric_limits<size_t>::max()};
  bool xpu_local_l3_autotune{true};
  size_t xpu_local_gm_size{0};
  int xpu_cluster_num{0};
  int xpu_sdnn_num{0};
  bool xpu_enable_multi_stream{false};
  // encoder config
  std::string multi_encoder_precision{"int16"};
  std::string compute_precision;
  bool multi_encoder_adaptive_seqlen{false};
  bool local_quant{false};
  int xpu_fc_autotune_level{0};
  std::string xpu_fc_autotune_file{""};
  bool xpu_fc_autotune_writeback{false};
  // kl1 : 0 per  tensor, 1 per batch, 2 per head
  // kl2 : 0 per tensor, else local_quant(per_16)
  int quant_post_dynamic_activation_method{0};
  int transformer_softmax_optimize_level{0};
  float quant_post_static_gelu_out_threshold{10.f};
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
  size_t api_l3_reserve{0};
};

}  // namespace lite
}  // namespace paddle
