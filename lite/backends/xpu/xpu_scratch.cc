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

  XPU_CALL(xpu_set_device(devid_));
  void* xpu_stream = TargetWrapperXPU::get_xpu_stream();
  VLOG(3) << "thread 0x" << std::hex << std::this_thread::get_id()
          << " set context xpu stream: " << xpu_stream;
  xpu_stream_ = xpu_stream;
  XPU_CALL(xpu_wait(xpu_stream_));
  XPU_CALL(xpu_free(addr_));
  addr_ = nullptr;

  addr_ = XPUMemory::Malloc(new_size);
  size_ = new_size;
}

XPUScratchPad::~XPUScratchPad() {
  XPU_CALL(xpu_set_device(devid_));
  XPU_CALL(xpu_wait(xpu_stream_));
  XPU_CALL(xpu_free(addr_));
  addr_ = nullptr;
  size_ = 0;
}

void* XPUMemory::Malloc(size_t size) {
  void* ptr{nullptr};
  if (size > 0) {
    int devid = -1;
    XPU_CALL(xpu_current_device(&devid));
    XPU_CALL(xpu_malloc(&ptr, size));
  }
  return ptr;
}

// Only used to free temporary buffer in the runtime.
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
  void* ptr = nullptr;
  if (size > 0) {
    ptr = XPUMemory::Malloc(size);
    CHECK(ptr) << "XPU Malloc Fail, Malloc Size is: " << size;
  }

  int devid = -1;
  XPU_CALL(xpu_current_device(&devid));
  void* xpu_stream = TargetWrapperXPU::get_xpu_stream();
  VLOG(6) << "thread 0x" << std::hex << std::this_thread::get_id()
          << " set context xpu stream: " << xpu_stream;
  return XPUScratchPadGuard(new XPUScratchPad(ptr, size, devid, xpu_stream));
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
