/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "fpga/common/fpga_common.h"
#include <algorithm>
#include <map>
#include "fpga/common/config.h"
#include "fpga/common/driver.h"

namespace paddle_mobile {
namespace fpga {

int16_t fp32_2_fp16(float fp32_num) {
  unsigned long tmp = *(unsigned long *)(&fp32_num);  // NOLINT
  auto t = (int16_t)(((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) |
                     (((tmp & 0x7f800000) >> 13) - (112 << 10)));
  if (tmp & 0x1000) {
    t++;  // roundoff
  }
  return t;
}

float fp16_2_fp32(int16_t fp16_num) {
  if (0 == fp16_num) {
    return 0;
  }
  int frac = (fp16_num & 0x3ff);
  int exp = ((fp16_num & 0x7c00) >> 10) + 112;
  int s = fp16_num & 0x8000;
  int tmp = 0;
  float fp32_num;
  tmp = s << 16 | exp << 23 | frac << 13;
  fp32_num = *(float *)&tmp;  // NOLINT
  return fp32_num;
}

static std::map<void *, size_t> memory_map;

int open_device() {
  int ret = driver::open_device_driver();
  return ret;
}

int close_device() {
  int ret = driver::close_device_driver();
  return ret;
}

void *fpga_malloc(size_t size) {
  static uint64_t counter = 0;
#ifdef PADDLE_MOBILE_ZU5
  auto ptr = driver::fpga_malloc_driver(size);
#else
  auto ptr = malloc(size);
#endif
  counter += size;
  memory_map.insert(std::make_pair(ptr, size));
  //  DLOG << "Address: " << ptr << ", " << size << " bytes allocated. Total "
  //       << counter << " bytes";
  return ptr;
}

void fpga_free(void *ptr) {
  static uint64_t counter = 0;
  size_t size = 0;
  auto iter = memory_map.find(ptr);  // std::map<void *, size_t>::iterator
  if (iter != memory_map.end()) {
    size = iter->second;
    memory_map.erase(iter);
#ifdef PADDLE_MOBILE_ZU5
    driver::fpga_free_driver(ptr);
#else
    free(ptr);
#endif
    counter += size;
    //    DLOG << "Address: " << ptr << ", " << size << " bytes freed. Total "
    //         << counter << " bytes";
  } else {
    DLOG << "Invalid pointer";
  }
}
void fpga_copy(void *dest, const void *src, size_t num) {
#ifdef PADDLE_MOBILE_ZU5
  // driver::fpga_copy_driver(dest, src, num);
  memcpy(dest, src, num);
#else
  memcpy(dest, src, num);
#endif
}

int fpga_flush(void *address, size_t size) {
#ifdef PADDLE_MOBILE_ZU5
  return driver::fpga_flush_driver(address, size);
#else
  return 0;
#endif
}
int fpga_invalidate(void *address, size_t size) {
#ifdef PADDLE_MOBILE_ZU5
  return driver::fpga_invalidate_driver(address, size);
#else
  return 0;
#endif
}
uint64_t vaddr_to_paddr(void *address) {
#ifdef PADDLE_MOBILE_ZU5
  return driver::vaddr_to_paddr(address);
#else
  return 0;
#endif
}
}  // namespace fpga
}  // namespace paddle_mobile
