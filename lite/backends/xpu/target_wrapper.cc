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

void* TargetWrapperXPU::Malloc(size_t size) {
  void* ptr{nullptr};
  if (size > 0) {
    XPU_CALL(xpu_malloc(&ptr, size));
  }
  return ptr;
}

void TargetWrapperXPU::Free(void* ptr) { XPU_CALL(xpu_free(ptr)); }

void TargetWrapperXPU::MemcpySync(void* dst,
                                  const void* src,
                                  size_t size,
                                  IoDirection dir) {
  switch (dir) {
    case IoDirection::HtoD:
      XPU_CALL(xpu_memcpy(dst, src, size, XPU_HOST_TO_DEVICE));
      break;
    case IoDirection::DtoH:
      // TODO(weihaoji): remove xpu_wait
      XPU_CALL(xpu_wait());
      XPU_CALL(xpu_memcpy(dst, src, size, XPU_DEVICE_TO_HOST));
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

XPUScratchPadGuard TargetWrapperXPU::MallocScratchPad(size_t size) {
  void* ptr = TargetWrapperXPU::Malloc(size);
  CHECK(ptr != nullptr) << "XPU Malloc Fail, Malloc Size is: " << size;
  return XPUScratchPadGuard(new XPUScratchPad(ptr, size));
}

void TargetWrapperXPU::MallocL3Cache(xdnn::Context* tls_raw_ctx,
                                     size_t l3_size,
                                     bool locked) {
  if (!TargetWrapperXPU::IsSharedL3Created() && shared_l3_size > 0) {
    mutex_l3_.lock();
    if (!TargetWrapperXPU::IsSharedL3Created()) {
      XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&shared_l3_ptr_),
                          shared_l3_size,
                          XPU_MEM_L3));
    }
    mutex_l3_.unlock();
  }
  if (!locked && l3_size > 0) {
    VLOG(3) << "Try To Malloc Local L3 Cache Size is" << l3_size;
    void* local_l3_ptr = nullptr;
    int ret = xpu_malloc(
        reinterpret_cast<void**>(&local_l3_ptr), l3_size, XPU_MEM_L3);
    if (ret != 0) {
      VLOG(3) << "No Enough L3 Cache For Current Predictor.";
    } else {
      ret = tls_raw_ctx->_l3_mgr.set(local_l3_ptr, l3_size);
      if (ret != 0) {
        LOG(WARNING) << "XPU L3 Mgr Init Fail, Please Check Configuration.";
        XPU_CALL(xpu_free(local_l3_ptr));
      } else {
        VLOG(3) << "Success!";
      }
    }
  } else if (TargetWrapperXPU::IsSharedL3Created()) {
    mutex_l3_.lock();
    XPU_CALL(tls_raw_ctx->_l3_mgr.set(shared_l3_ptr_, shared_l3_size));
  }
}

void TargetWrapperXPU::FreeL3Cache(xdnn::Context* tls_raw_ctx,
                                   size_t l3_size,
                                   bool locked) {
  if (!locked && l3_size > 0) {
    void* xpu_l3_ptr = tls_raw_ctx->_l3_mgr.get_ptr();
    if (xpu_l3_ptr != nullptr) {
      XPU_CALL(xpu_wait());
      XPU_CALL(xpu_free(xpu_l3_ptr))
      XPU_CALL(tls_raw_ctx->_l3_mgr.set(nullptr, 0));
    }
  } else if (TargetWrapperXPU::IsSharedL3Created()) {
    XPU_CALL(xpu_wait());
    XPU_CALL(tls_raw_ctx->_l3_mgr.set(nullptr, 0));
    mutex_l3_.unlock();
  }
}

size_t TargetWrapperXPU::shared_l3_size{0};
void* TargetWrapperXPU::shared_l3_ptr_{nullptr};
std::mutex TargetWrapperXPU::mutex_l3_;

}  // namespace lite
}  // namespace paddle
