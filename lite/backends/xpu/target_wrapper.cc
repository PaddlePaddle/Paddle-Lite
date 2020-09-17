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

  if (!is_l3_) {
    TargetWrapperXPU::Free(addr_);
    addr_ = TargetWrapperXPU::Malloc(new_size);
    size_ = new_size;
  } else {
    CHECK(false) << "Not supported if is_l3_ == true";
  }
}

void XPUScratchPadDeleter::operator()(XPUScratchPad* sp) const {
  if (!sp->is_l3_) {
    TargetWrapperXPU::Free(sp->addr_);
  }
  delete sp;
}

void* TargetWrapperXPU::Malloc(size_t size) {
  void* ptr{nullptr};
  XPU_CALL(xpu_malloc(&ptr, size));
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
      XPU_CALL(xpu_memcpy(dst, src, size, XPU_DEVICE_TO_HOST));
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

XPUScratchPadGuard TargetWrapperXPU::MallocScratchPad(size_t size,
                                                      bool use_l3) {
  void* ptr{nullptr};
  if (use_l3) {
    ptr = xdnn::alloc_workspace(GetRawContext(), size);
  } else {
    ptr = TargetWrapperXPU::Malloc(size);
  }
  CHECK(ptr != nullptr) << "size = " << size << ", use_l3 = " << use_l3;
  return XPUScratchPadGuard(new XPUScratchPad(ptr, size, use_l3));
}

std::string TargetWrapperXPU::multi_encoder_precision;  // NOLINT
int TargetWrapperXPU::workspace_l3_size_per_thread{0};
LITE_THREAD_LOCAL xdnn::Context* TargetWrapperXPU::tls_raw_ctx_{nullptr};

}  // namespace lite
}  // namespace paddle
