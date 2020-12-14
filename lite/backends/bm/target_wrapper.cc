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
#include "lite/backends/bm/target_wrapper.h"
#include <bmcompiler_if.h>
#include <bmlib_runtime.h>
#include <utility>

namespace paddle {
namespace lite {

int TargetWrapperBM::device_id_ = 0;
std::map<int, void*> TargetWrapperBM::bm_hds_;

size_t TargetWrapperBM::num_devices() {
  int count = 1;
#if !defined(BM_SAVE_UMODEL) && !defined(BM_SAVE_BMODEL)
  bm_status_t ret = bm_dev_getcount(&count);
  CHECK_EQ(ret, BM_SUCCESS) << "Failed with error code: "
                            << static_cast<int>(ret);
#endif
  return count;
}

int TargetWrapperBM::GetDevice() { return device_id_; }
void TargetWrapperBM::SetDevice(int id) {
  if (id < 0 || (size_t)id >= num_devices()) {
    LOG(FATAL) << "Failed with invalid device id " << id;
  }
  device_id_ = id;
#if !defined(BM_SAVE_UMODEL) && !defined(BM_SAVE_BMODEL)
  if (bm_hds_.find(id) == bm_hds_.end()) {
    bm_handle_t bm_handle;
    bm_status_t ret = bm_dev_request(&bm_handle, id);
    CHECK_EQ(ret, BM_SUCCESS) << "Failed with error code: "
                              << static_cast<int>(ret);
    bm_hds_.insert(std::pair<int, bm_handle_t>(id, bm_handle));
  }
#endif
  return;
}

void* TargetWrapperBM::GetHandle() {
  if (bm_hds_.find(device_id_) == bm_hds_.end()) {
    LOG(FATAL) << "device not initialized " << device_id_;
  }
  return bm_hds_.at(device_id_);
}

void* TargetWrapperBM::Malloc(size_t size) {
  void* ptr{};

  if (bm_hds_.find(device_id_) == bm_hds_.end()) {
    SetDevice(device_id_);
  }

  bm_handle_t bm_handle = static_cast<bm_handle_t>(bm_hds_.at(device_id_));
  bm_device_mem_t* p_mem =
      reinterpret_cast<bm_device_mem_t*>(malloc(sizeof(bm_device_mem_t)));
  bm_malloc_device_byte(bm_handle, p_mem, size);
  ptr = reinterpret_cast<void*>(p_mem);
  return ptr;
}

void TargetWrapperBM::Free(void* ptr) {
  if (ptr != NULL) {
    bm_handle_t bm_handle = static_cast<bm_handle_t>(bm_hds_.at(device_id_));
    bm_device_mem_t* mem = static_cast<bm_device_mem_t*>(ptr);
    bm_free_device(bm_handle, *mem);
    free(ptr);
  }
  return;
}

void TargetWrapperBM::MemcpySync(void* dst,
                                 const void* src,
                                 size_t size,
                                 IoDirection dir) {
  if (bm_hds_.find(device_id_) == bm_hds_.end()) {
    return;
  }

  bm_handle_t bm_handle = static_cast<bm_handle_t>(bm_hds_.at(device_id_));
  bm_device_mem_t* pmem{};
  const bm_device_mem_t* pcst_mem{};

  switch (dir) {
    case IoDirection::HtoD:
      pmem = static_cast<bm_device_mem_t*>(dst);
      bm_memcpy_s2d_partial_offset(
          bm_handle, *pmem, const_cast<void*>(src), size, 0);
      break;
    case IoDirection::DtoH:
      pcst_mem = static_cast<const bm_device_mem_t*>(src);
      bm_memcpy_d2s_partial_offset(
          bm_handle, reinterpret_cast<void*>(dst), *pcst_mem, size, 0);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
      break;
  }
  return;
}

}  // namespace lite
}  // namespace paddle
