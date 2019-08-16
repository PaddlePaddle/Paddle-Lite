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
#include <utility>
#include "fpga/common/config.h"
#include "fpga/common/driver.h"

namespace paddle_mobile {
namespace fpga {

int16_t fp32_2_fp16(float fp32_num) {
  int32_t tmp = *(reinterpret_cast<int32_t *>(&fp32_num));
  int16_t se_fp32 = (tmp >> 23) & 0x1ff;
  int32_t m_fp32 = tmp & 0x007fffff;
  int16_t se_fp16 = 0;
  int16_t m_fp16 = 0;

  if (se_fp32 < 103) {
    se_fp16 = 0x0000;
    m_fp16 = m_fp32 >> 24;
  } else if (se_fp32 < 113) {
    se_fp16 = (0x0400 >> (113 - se_fp32));
    m_fp16 = m_fp32 >> (126 - se_fp32);
  } else if (se_fp32 <= 142) {
    se_fp16 = (se_fp32 - 112) << 10;
    m_fp16 = m_fp32 >> 13;
  } else if (se_fp32 < 255) {
    se_fp16 = 0x7C00;
    m_fp16 = m_fp32 >> 24;
  } else if (se_fp32 == 255) {
    se_fp16 = 0x7C00;
    m_fp16 = m_fp32 >> 13;
  } else if (se_fp32 < 359) {
    se_fp16 = 0x8000;
    m_fp16 = m_fp32 >> 24;
  } else if (se_fp32 < 369) {
    se_fp16 = (0x0400 >> (369 - se_fp32)) | 0x8000;
    m_fp16 = m_fp32 >> (382 - se_fp32);
  } else if (se_fp32 <= 398) {
    se_fp16 = ((se_fp32 - 368) << 10) | 0x8000;
    m_fp16 = m_fp32 >> 13;
  } else if (se_fp32 < 511) {
    se_fp16 = 0x7C00;
    m_fp16 = m_fp32 >> 24;
  } else {
    se_fp16 = 0x7C00;
    m_fp16 = m_fp32 >> 13;
  }
  int16_t result = se_fp16 + m_fp16;
  return result;
}

int32_t convertmantissa(int32_t i) {
  int32_t m = i << 13;
  int32_t e = 0;
  while (!(m & 0x00800000)) {
    e -= 0x00800000;
    m <<= 1;
  }
  m &= ~0x00800000;
  e += 0x38800000;
  return m | e;
}

float fp16_2_fp32(int16_t fp16_num) {
  int16_t se_fp16 = (fp16_num >> 10) & 0x3f;
  int16_t m_fp16 = fp16_num & 0x3ff;
  int32_t e_fp32 = 0;
  int16_t offset = 0;
  int32_t m_fp32 = 0;
  if (se_fp16 == 0) {
    e_fp32 = 0;
    offset = 0;
  } else if (se_fp16 < 31) {
    e_fp32 = se_fp16 << 23;
    offset = 1024;
  } else if (se_fp16 == 31) {
    e_fp32 = 0x47800000;
    offset = 1024;
  } else if (se_fp16 == 32) {
    e_fp32 = 0x80000000;
    offset = 0;
  } else if (se_fp16 < 63) {
    e_fp32 = 0x80000000 + ((se_fp16 - 32) << 23);
    offset = 1024;
  } else {  // se_fp16 == 63
    e_fp32 = 0xC7800000;
    offset = 1024;
  }
  int16_t a = offset + m_fp16;
  if (a == 0) {
    m_fp32 = 0;
  } else if (a < 1024) {
    int32_t tmp = a;
    m_fp32 = convertmantissa(tmp);
  } else {
    int32_t tmp = a - 1024;
    m_fp32 = 0x38000000 + (tmp << 13);
  }

  int32_t tmp = e_fp32 + m_fp32;
  float fp32_num = *(reinterpret_cast<float *>(&tmp));
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
  if (size <= 0) {
    size = 1;
  }
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
  if (ptr == nullptr) {
    return;
  }
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
    DLOG << "Address: " << ptr << "  Invalid pointer";
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
  return driver::vaddr_to_paddr_driver(address);
#else
  return 0;
#endif
}

uint32_t paddle_mobile_version() {
  uint32_t v_master = 52;
  uint32_t v_slave = 52;

  uint32_t first = 1, second = 2, fourth_master = 1, fourth_slave = 1;
  uint32_t master = first << 24 | second << 16 | v_master << 8 | fourth_master;
  uint32_t slave = first << 24 | second << 16 | v_slave << 8 | fourth_slave;

  return slave;
}

}  // namespace fpga
}  // namespace paddle_mobile
