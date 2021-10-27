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

void XPUScratchPad::Reserve(size_t new_size) {
  if (new_size <= size_) {
    return;
  }
  TargetWrapperXPU::Free(addr_);
  addr_ = TargetWrapperXPU::Malloc(new_size);
  size_ = new_size;
}

void XPUScratchPadDeleter::operator()(XPUScratchPad* sp) const {
  TargetWrapperXPU::Free(sp->addr_);
  delete sp;
}

XPUL3CacheBlock* TargetWrapperXPU::CreateL3CacheBlock() {
  l3_block_dict.push_back(new XPUL3CacheBlock());
  return l3_block_dict.back();
}

void* TargetWrapperXPU::Malloc(size_t size) {
  void* ptr{nullptr};
  if (size > 0) {
    XPU_CALL(xpu_malloc(&ptr, size));
  }
  return ptr;
}

void TargetWrapperXPU::Free(void* ptr) {
  XPU_CALL(xpu_wait());
  XPU_CALL(xpu_free(ptr));
}

void TargetWrapperXPU::MemcpySync(void* dst,
                                  const void* src,
                                  size_t size,
                                  IoDirection dir) {
  switch (dir) {
    case IoDirection::HtoD:
      XPU_CALL(xpu_wait());
      XPU_CALL(xpu_memcpy(dst, src, size, XPU_HOST_TO_DEVICE));
      break;
    case IoDirection::DtoH:
      XPU_CALL(xpu_wait());
      XPU_CALL(xpu_memcpy(dst, src, size, XPU_DEVICE_TO_HOST));
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

XPUScratchPadGuard TargetWrapperXPU::MallocScratchPad(size_t size) {
  void* ptr = TargetWrapperXPU::Malloc(size);
  CHECK(ptr) << "XPU Malloc Fail, Malloc Size is: " << size;
  return XPUScratchPadGuard(new XPUScratchPad(ptr, size));
}

void TargetWrapperXPU::ScatterL3Cache(
    void* l3_ptr,
    size_t l3_size,
    const std::vector<std::vector<int64_t>>& query_shape) {
  if (l3_size == 0 || l3_ptr == nullptr) {
    LOG(FATAL) << "Unable to scatter l3 cache, l3 cache size is: " << l3_size
               << ", l3 addr is: " << l3_ptr;
  }
  CHECK(l3_planner_);
  size_t total_block_l3_size = 0, xdnn_ctx_l3_size = 0;
  l3_planner_->set_current_query_shape(query_shape, l3_size);
  std::vector<size_t>* plan = l3_planner_->get_current_plan();
  if (plan == nullptr) {
    XPU_CALL(tls_raw_ctx_->_l3_mgr.set(l3_ptr, l3_size));
    VLOG(3) << "XDNN CTX L3 Size is " << tls_raw_ctx_->_l3_mgr.get_size()
            << ", Remain L3 Size for Lite is " << 0;
  } else {
    CHECK_EQ(plan->size(), l3_block_dict.size() + 1);
    xdnn_ctx_l3_size = plan->back();
    for (size_t i = 0; i < l3_block_dict.size(); i++) {
      size_t cur_block_size = plan->data()[i];
      if (cur_block_size > 0) {
        l3_block_dict[i]->set(
            reinterpret_cast<int8_t*>(l3_ptr) + total_block_l3_size,
            cur_block_size);
        total_block_l3_size += cur_block_size;
      }
    }
    if (xdnn_ctx_l3_size > 0) {
      XPU_CALL(tls_raw_ctx_->_l3_mgr.set(
          reinterpret_cast<int8_t*>(l3_ptr) + l3_size - xdnn_ctx_l3_size,
          xdnn_ctx_l3_size));
    }
    VLOG(3) << "XDNN CTX L3 Size is " << tls_raw_ctx_->_l3_mgr.get_size()
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
  if (local_l3_size != 0) {
    // malloc local_l3
    VLOG(3) << "Try To Malloc Local L3 Cache Size is" << local_l3_size;
    int ret = xpu_malloc(
        reinterpret_cast<void**>(&local_l3_ptr_), local_l3_size, XPU_MEM_L3);
    if (ret != 0) {
      VLOG(3) << "No Enough L3 Cache For Current Predictor.";
      local_l3_ptr_ = nullptr;
    } else {
      VLOG(3) << "Success!";
      TargetWrapperXPU::ScatterL3Cache(
          local_l3_ptr_, local_l3_size, query_shape);
    }
  } else if (need_l3_mutex && TargetWrapperXPU::IsSharedL3Created()) {
    // lock and use shared_l3
    mutex_l3_.lock();
    TargetWrapperXPU::ScatterL3Cache(
        shared_l3_ptr_, shared_l3_size, query_shape);
  }
}

void TargetWrapperXPU::FreeL3Cache() {
  if (local_l3_size != 0) {
    if (local_l3_ptr_ != nullptr) {
      TargetWrapperXPU::Free(local_l3_ptr_);
      local_l3_ptr_ = nullptr;
      XPU_CALL(tls_raw_ctx_->_l3_mgr.set(nullptr, 0));
    }
    l3_planner_->run_autotune(l3_block_dict, local_l3_size);
  } else if (need_l3_mutex && TargetWrapperXPU::IsSharedL3Created()) {
    XPU_CALL(xpu_wait());
    XPU_CALL(tls_raw_ctx_->_l3_mgr.set(nullptr, 0));
    mutex_l3_.unlock();
    l3_planner_->run_autotune(l3_block_dict, shared_l3_size);
  }
  for (size_t i = 0; i < l3_block_dict.size(); i++) {
    l3_block_dict[i]->clear();
  }
}

// xpu context
LITE_THREAD_LOCAL xdnn::Context* TargetWrapperXPU::tls_raw_ctx_{nullptr};
// multi encoder config
LITE_THREAD_LOCAL std::string
    TargetWrapperXPU::multi_encoder_precision;  // NOLINT
LITE_THREAD_LOCAL bool TargetWrapperXPU::multi_encoder_adaptive_seqlen{false};
// conv autotune config
LITE_THREAD_LOCAL bool TargetWrapperXPU::conv_autotune{false};
LITE_THREAD_LOCAL std::string TargetWrapperXPU::conv_autotune_file;
// l3 cache config
LITE_THREAD_LOCAL bool TargetWrapperXPU::need_l3_mutex{false};
LITE_THREAD_LOCAL size_t TargetWrapperXPU::local_l3_size{
    std::numeric_limits<size_t>::max()};
LITE_THREAD_LOCAL size_t TargetWrapperXPU::local_gm_size{
    0x4000000};  // 64 * 1024 * 1024
LITE_THREAD_LOCAL void* TargetWrapperXPU::local_l3_ptr_{nullptr};
void* TargetWrapperXPU::shared_l3_ptr_{nullptr};
size_t TargetWrapperXPU::shared_l3_size{0};
LITE_THREAD_LOCAL std::vector<XPUL3CacheBlock*> TargetWrapperXPU::l3_block_dict;
// l3 mutex
std::mutex TargetWrapperXPU::mutex_l3_;
// l3 planner
LITE_THREAD_LOCAL XPUL3Planner* TargetWrapperXPU::l3_planner_{nullptr};

}  // namespace lite
}  // namespace paddle
