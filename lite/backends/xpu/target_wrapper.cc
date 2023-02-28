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

#include "lite/backends/xpu/target_wrapper.h"
#include <limits>
#include <vector>
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

XPUL3CacheBlock* TargetWrapperXPU::CreateL3CacheBlock() {
  if (xpu_runtime_ptr == nullptr) {
    return nullptr;
  }

  xpu_runtime_ptr->xpu_l3_block_dict.push_back(new XPUL3CacheBlock());
  return xpu_runtime_ptr->xpu_l3_block_dict.back();
}

void TargetWrapperXPU::MemcpySync(void* dst,
                                  const void* src,
                                  size_t size,
                                  IoDirection dir) {
  switch (dir) {
    case IoDirection::HtoD:
      XPUMemory::MemcpyHtoDSync(dst, src, size);
      break;
    case IoDirection::DtoH:
      XPUMemory::MemcpyDtoHSync(dst, src, size);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

template <typename Tcpu, typename Txpu>
XPUQuantData TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight(
    const Tcpu* cpu_data,
    const DDimLite& dims,
    bool data_transpose,
    size_t max_ptr_len) {
  return xpu_runtime_ptr->quantizer.quant<Tcpu, Txpu>(
      cpu_data, dims, data_transpose, max_ptr_len);
}

void TargetWrapperXPU::ScatterL3Cache(
    void* l3_ptr,
    size_t l3_size,
    const std::vector<std::vector<int64_t>>& query_shape) {
  if (l3_size == 0 || l3_ptr == nullptr) {
    LOG(FATAL) << "Unable to scatter l3 cache, l3 cache size is: " << l3_size
               << ", l3 addr is: " << l3_ptr;
  }
  CHECK(xpu_runtime_ptr->xpu_l3_planner);
  size_t total_block_l3_size = 0, xdnn_ctx_l3_size = 0;
  xpu_runtime_ptr->xpu_l3_planner->set_current_query_shape(query_shape,
                                                           l3_size);
  std::vector<size_t>* plan =
      xpu_runtime_ptr->xpu_l3_planner->get_current_plan();
  if (plan == nullptr) {
    XPU_CALL(xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext()->_l3_mgr.set(
        l3_ptr, l3_size));
    VLOG(3) << "XDNN CTX L3 Size is "
            << xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext()
                   ->_l3_mgr.get_size()
            << ", Remain L3 Size for Lite is " << 0;
  } else {
    CHECK_EQ(plan->size(), xpu_runtime_ptr->xpu_l3_block_dict.size() + 1);
    xdnn_ctx_l3_size = plan->back();
    for (size_t i = 0; i < xpu_runtime_ptr->xpu_l3_block_dict.size(); i++) {
      size_t cur_block_size = plan->data()[i];
      if (cur_block_size > 0) {
        xpu_runtime_ptr->xpu_l3_block_dict[i]->set(
            reinterpret_cast<int8_t*>(l3_ptr) + total_block_l3_size,
            cur_block_size);
        total_block_l3_size += cur_block_size;
      }
    }
    if (xdnn_ctx_l3_size > 0) {
      XPU_CALL(xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext()->_l3_mgr.set(
          reinterpret_cast<int8_t*>(l3_ptr) + l3_size - xdnn_ctx_l3_size,
          xdnn_ctx_l3_size));
    }
    VLOG(3) << "XDNN CTX L3 Size is "
            << xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext()
                   ->_l3_mgr.get_size()
            << ", Remain L3 Size for Lite is " << total_block_l3_size;
  }
}

void TargetWrapperXPU::MallocL3Cache(
    const std::vector<std::vector<int64_t>>& query_shape) {
  TargetWrapperXPU::GetRawContext();
  // malloc shared l3
  if (!TargetWrapperXPU::IsSharedL3Created() && shared_l3_size > 0) {
    mutex_l3_.lock();
    if (!TargetWrapperXPU::IsSharedL3Created()) {
      XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&shared_l3_ptr_),
                          shared_l3_size,
                          XPU_MEM_L3));
    }
    mutex_l3_.unlock();
  }
  if (xpu_runtime_ptr->xpu_local_l3_size != 0) {
    // malloc local_l3
    VLOG(3) << "Try To Malloc Local L3 Cache Size is"
            << xpu_runtime_ptr->xpu_local_l3_size;
    int ret = xpu_malloc(
        reinterpret_cast<void**>(&(xpu_runtime_ptr->xpu_local_l3_ptr)),
        xpu_runtime_ptr->xpu_local_l3_size,
        XPU_MEM_L3);
    if (ret != 0) {
      VLOG(3) << "No Enough L3 Cache For Current Predictor.";
      xpu_runtime_ptr->xpu_local_l3_ptr = nullptr;
    } else {
      VLOG(3) << "Success!";
      TargetWrapperXPU::ScatterL3Cache(xpu_runtime_ptr->xpu_local_l3_ptr,
                                       xpu_runtime_ptr->xpu_local_l3_size,
                                       query_shape);
    }
  } else if (need_l3_mutex && TargetWrapperXPU::IsSharedL3Created()) {
    // lock and use shared_l3
    mutex_l3_.lock();
    TargetWrapperXPU::ScatterL3Cache(
        shared_l3_ptr_, shared_l3_size, query_shape);
  }
}

void TargetWrapperXPU::FreeL3Cache() {
  if (xpu_runtime_ptr->xpu_local_l3_size != 0) {
    if (xpu_runtime_ptr->xpu_local_l3_ptr != nullptr) {
      TargetWrapperXPU::Free(xpu_runtime_ptr->xpu_local_l3_ptr);
      xpu_runtime_ptr->xpu_local_l3_ptr = nullptr;
      XPU_CALL(xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext()->_l3_mgr.set(
          nullptr, 0));
    }
    if (xpu_runtime_ptr->xpu_local_l3_autotune) {
      xpu_runtime_ptr->xpu_l3_planner->run_autotune(
          xpu_runtime_ptr->xpu_l3_block_dict,
          xpu_runtime_ptr->xpu_local_l3_size);
    }
  } else if (need_l3_mutex && TargetWrapperXPU::IsSharedL3Created()) {
    XPU_CALL(xpu_wait(TargetWrapperXPU::get_xpu_stream()));
    XPU_CALL(xpu_runtime_ptr->xpu_tls_raw_ctx->GetXDNNContext()->_l3_mgr.set(
        nullptr, 0));
    mutex_l3_.unlock();
    if (xpu_runtime_ptr->xpu_local_l3_autotune) {
      xpu_runtime_ptr->xpu_l3_planner->run_autotune(
          xpu_runtime_ptr->xpu_l3_block_dict, shared_l3_size);
    }
  }
  for (size_t i = 0; i < xpu_runtime_ptr->xpu_l3_block_dict.size(); i++) {
    xpu_runtime_ptr->xpu_l3_block_dict[i]->clear();
  }
}

template XPUQuantData
TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, float>(
    const float*, const DDimLite&, bool, size_t);
template XPUQuantData
TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
    const float*, const DDimLite&, bool, size_t);
template XPUQuantData
TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int8_t>(
    const float*, const DDimLite&, bool, size_t);
template XPUQuantData
TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int8_t, int8_t>(
    const int8_t*, const DDimLite&, bool, size_t);
template XPUQuantData
TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int16_t, int16_t>(
    const int16_t*, const DDimLite&, bool, size_t);

// l3 cache config
LITE_THREAD_LOCAL bool TargetWrapperXPU::need_l3_mutex{false};
void* TargetWrapperXPU::shared_l3_ptr_{nullptr};
size_t TargetWrapperXPU::shared_l3_size{0};
// l3 mutex
std::mutex TargetWrapperXPU::mutex_l3_;
// xpu quantizer
LITE_THREAD_LOCAL XPURunTimeOption* TargetWrapperXPU::xpu_runtime_ptr{nullptr};
}  // namespace lite
}  // namespace paddle
