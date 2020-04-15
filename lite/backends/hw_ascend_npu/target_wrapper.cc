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

#include "lite/backends/hw_ascend_npu/target_wrapper.h"
#include <acl/acl.h>
#include <glog/logging.h>

namespace paddle {
namespace lite {

void* TargetWrapperHWAscendNPU::Malloc(size_t size) {
  void* ptr{nullptr};
  if (ACL_ERROR_NONE != aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY)) {
    LOG(ERROR) << "[HWAscendNPU]: Allocate memory from device failed";
    ptr = nullptr;
  }
  return ptr;
}

void TargetWrapperHWAscendNPU::Free(void* ptr) { aclrtFree(ptr); }

void TargetWrapperHWAscendNPU::MemcpySync(void* dst,
                                          const void* src,
                                          size_t size,
                                          IoDirection dir) {
  switch (dir) {
    case IoDirection::HtoD:
      aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
      break;
    case IoDirection::DtoH:
      aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

}  // namespace lite
}  // namespace paddle
