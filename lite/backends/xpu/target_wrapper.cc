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
#include <fcntl.h>
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

void TargetWrapperXPU::LockXPU(int dev_no) {
  // TODO(zhupengyang): support multi-xpu later
  if (reentrant_ == 0) {
    struct flock f_lock;
    f_lock.l_whence = 0;
    f_lock.l_len = 0;
    f_lock.l_type = F_WRLCK;

    std::string buf = "/opt/xpu_lock" + std::to_string(dev_no);
    CHECK_EQ(xpu_lock_fd_, -1) << "file: " << buf << " has been opened before.";
    xpu_lock_fd_ = open(buf.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH);
    CHECK_GT(xpu_lock_fd_, -1) << "open " << buf << " failed: " << xpu_lock_fd_;

    fcntl(xpu_lock_fd_, F_SETLKW, &f_lock);
  }
  reentrant_++;
}

void TargetWrapperXPU::ReleaseXPU(int dev_no) {
  if (xpu_lock_fd_ < 0 && reentrant_ == 0) return;

  if (reentrant_ == 1) {
    struct flock f_lock;
    f_lock.l_whence = 0;
    f_lock.l_len = 0;
    f_lock.l_type = F_UNLCK;
    CHECK_GT(xpu_lock_fd_, -1) << "File may be closed before.";
    fcntl(xpu_lock_fd_, F_SETLKW, &f_lock);
    close(xpu_lock_fd_);
    xpu_lock_fd_ = -1;
  }
  reentrant_--;
}

std::string TargetWrapperXPU::multi_encoder_precision;  // NOLINT
int TargetWrapperXPU::workspace_l3_size_per_thread{0};
LITE_THREAD_LOCAL xdnn::Context* TargetWrapperXPU::tls_raw_ctx_{nullptr};
int TargetWrapperXPU::reentrant_ = 0;
int TargetWrapperXPU::xpu_lock_fd_ = -1;

}  // namespace lite
}  // namespace paddle
