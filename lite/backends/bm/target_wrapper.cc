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
#include <map>
#include "lite/backends/bm/target_wrapper.h"
#include "bmlib_runtime.h"
#include "bmcompiler_if.h"

namespace paddle {
namespace lite {

static int g_current_device_id = 0;
static std::map<int, bm_handle_t> g_bm_handles;

size_t TargetWrapperBM::num_devices() {
  int count = 0;
  bm_dev_getcount(&count);
  return count;
}

void TargetWrapperBM::SetDevice(int id) {
  g_current_device_id = id;

  if (g_bm_handles.find(id) == g_bm_handles.end()) {
    bm_handle_t bm_handle;
    bm_status_t ret = bm_dev_request(&bm_handle, id);
    CHECK_EQ(ret, BM_SUCCESS) << "Failed with error code: " << (int)ret;
    g_bm_handles.insert(std::pair<int, bm_handle_t>(id, bm_handle));
  }
  return;
}

void* TargetWrapperBM::Malloc(size_t size) {
  void* ptr{};

  if (g_bm_handles.find(g_current_device_id) == g_bm_handles.end()) {
      SetDevice(g_current_device_id);
  } 

  bm_handle_t bm_handle = g_bm_handles.at(g_current_device_id);
  bm_device_mem_t* p_mem = (bm_device_mem_t*)malloc(sizeof(bm_device_mem_t));
  bm_malloc_device_byte(bm_handle, p_mem, size);
  ptr = (void*)p_mem;
  return ptr;
}

void TargetWrapperBM::Free(void* ptr) {
  if (ptr != NULL) {
    bm_handle_t bm_handle = g_bm_handles.at(g_current_device_id);
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
  if (g_bm_handles.find(g_current_device_id) == g_bm_handles.end()){
    return;
  }

  bm_handle_t bm_handle = g_bm_handles.at(g_current_device_id);
  bm_device_mem_t* pmem{};
  const bm_device_mem_t* pcst_mem{};

  switch (dir) {
    case IoDirection::HtoD:
      pmem = static_cast<bm_device_mem_t*>(dst);
      bm_memcpy_s2d_partial_offset(bm_handle, *pmem, (void*)src, size, 0);
      break;
    case IoDirection::DtoH:
      pcst_mem = static_cast<const bm_device_mem_t*>(src);
      bm_memcpy_d2s_partial_offset(bm_handle, (void*)(dst), *pcst_mem, size, 0);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
      break;
  }
  return;
}
    
}  // namespace lite
}  // namespace paddle
