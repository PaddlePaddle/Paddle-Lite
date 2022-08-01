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

#include "lite/backends/xpu/xpu_scratch.h"
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {

void XPUScratchPad::Reserve(size_t new_size) {
  if (new_size <= size_) {
    return;
  }
  XPUMemory::Free(addr_);
  addr_ = XPUMemory::Malloc(new_size);
  size_ = new_size;
}

void XPUScratchPadDeleter::operator()(XPUScratchPad* sp) const {
  XPUMemory::Free(sp->addr_);
  sp->addr_ = nullptr;
  sp->size_ = 0;
  delete sp;
}

void* XPUMemory::Malloc(size_t size) {
  void* ptr{nullptr};
  if (size > 0) {
    XPU_CALL(xpu_malloc(&ptr, size));
  }
  return ptr;
}

void XPUMemory::Free(void* ptr) {
  XPU_CALL(xpu_wait(TargetWrapperXPU::get_xpu_stream()));
  XPU_CALL(xpu_free(ptr));
  ptr = nullptr;
}

void XPUMemory::MemcpyHtoDSync(void* dst, const void* src, size_t size) {
  XPU_CALL(xpu_wait(TargetWrapperXPU::get_xpu_stream()));
  XPU_CALL(xpu_memcpy(dst, src, size, XPU_HOST_TO_DEVICE));
}

void XPUMemory::MemcpyDtoHSync(void* dst, const void* src, size_t size) {
  XPU_CALL(xpu_wait(TargetWrapperXPU::get_xpu_stream()));
  XPU_CALL(xpu_memcpy(dst, src, size, XPU_DEVICE_TO_HOST));
}

XPUScratchPadGuard XPUMemory::MallocScratchPad(size_t size) {
  void* ptr = XPUMemory::Malloc(size);
  CHECK(ptr) << "XPU Malloc Fail, Malloc Size is: " << size;
  return XPUScratchPadGuard(new XPUScratchPad(ptr, size));
}

int XPUMemory::get_max_ptr_size() {
  int dev_id = 0;
  uint64_t dev_attr = 0;
  XPU_CALL(xpu_device_get_attr(&dev_attr, XPUATTR_MODEL, dev_id));

  if (dev_attr == R200) {
    return 6;
  } else if (dev_attr == K100 || dev_attr == K200) {
    return 4;
  }
  return 6;
}

}  // namespace lite
}  // namespace paddle
