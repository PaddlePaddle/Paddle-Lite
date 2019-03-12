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

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

#include "common/enforce.h"
#include "fpga/common/driver.h"

namespace paddle_mobile {
namespace fpga {
namespace driver {
struct FPGA_INFO g_fpgainfo;

int open_drvdevice() {
  if (g_fpgainfo.fd_drv == -1) {
    g_fpgainfo.fd_drv = open(g_fpgainfo.drvdevice_path, O_RDWR);
  }
  return g_fpgainfo.fd_drv;
}

int open_memdevice() {
  if (g_fpgainfo.fd_mem == -1) {
    // g_fpgainfo.fd_mem = open(g_fpgainfo.memdevice_path, O_RDWR | O_DSYNC);
    g_fpgainfo.fd_mem = open(g_fpgainfo.memdevice_path, O_RDWR);
  }
  return g_fpgainfo.fd_mem;
}

void pl_reset() {
  // DLOG << "PL RESET";

  usleep(100 * 1000);
}

void setup_pe(struct pe_data_s *pe_data, struct fpga_pe *pe,
              char const *type_name, int pe_idx) {
  memset(pe, 0, sizeof(struct fpga_pe));

  pe->outer = pe_data;
  snprintf(pe->type_name, MAX_TYPE_NAME_LENTH, "%s", type_name);

  pe->status = IDLE;
  pe->interrupt_cnt = 0;
  pe_data->pes[pe_idx] = pe;
  pe_data->pe_num++;
}

void pl_init() {
  struct pe_data_s *pe_data = nullptr;

  pl_reset();

  pe_data = (struct pe_data_s *)malloc(sizeof(struct pe_data_s));
  if (pe_data == nullptr) {
    DLOG << "pe_data malloc error!";
    return;
  }
  memset(pe_data, 0, sizeof(struct pe_data_s));
  pthread_mutex_init(&pe_data->mutex, 0);

  setup_pe(pe_data, &pe_data->pe_conv, "CONV", PE_IDX_CONV);
  setup_pe(pe_data, &pe_data->pe_pooling, "POOLING", PE_IDX_POOLING);
  setup_pe(pe_data, &pe_data->pe_ew, "EW", PE_IDX_EW);
  setup_pe(pe_data, &pe_data->pe_bypass, "BYPASS", PE_IDX_BYPASS);

  g_fpgainfo.pe_data = pe_data;
}

void pl_destroy() {
  struct pe_data_s *pe_data = g_fpgainfo.pe_data;
  pthread_mutex_destroy(&pe_data->mutex);
  free(pe_data);
}

void pl_start() {
  struct pe_data_s *pe_data = g_fpgainfo.pe_data;

  pthread_mutex_unlock(&pe_data->mutex);
}

void pl_stop() {
  struct pe_data_s *pe_data = g_fpgainfo.pe_data;

  pthread_mutex_lock(&pe_data->mutex);
}

void pl_reinit() {
  struct pe_data_s *pe_data = g_fpgainfo.pe_data;
  struct fpga_pe *pe = nullptr;
  int i = 0;

  pl_stop();
  pl_reset();
  pl_start();

  for (i = 0; i < pe_data->pe_num; i++) {
    pe = pe_data->pes[i];
    pe->status = IDLE;
    pe->interrupt_cnt = 0;
  }

  pl_start();
}

int pl_get_status() { return 0; }

/*tmie单位us*/
int fpga_regpoll(uint64_t reg, uint64_t val, int time) {
  uint64_t i = 0;
  /*timeout精确性待确认*/
  int64_t timeout = time * 6;

  for (i = 0; i < timeout; i++) {
    if (val == reg_readq(reg)) {
      break;
    }
  }

  if (i < timeout) {
    return 0;
  } else {
    return -1;
  }
}

void memory_release(struct fpga_memory *memory) {
  void *ptr = nullptr;

  /*unmap memory*/
  std::map<void *, size_t> map = g_fpgainfo.fpga_addr2size_map;
  std::map<void *, size_t>::iterator iter;
  for (iter = map.begin(); iter != map.end(); iter++) {
    fpga_free_driver(ptr);
  }
}

uint64_t vaddr_to_paddr_driver(void *address) {
  uint64_t paddr = 0;
  auto iter = g_fpgainfo.fpga_vaddr2paddr_map.find(address);
  if (iter != g_fpgainfo.fpga_vaddr2paddr_map.end()) {
    paddr = iter->second;
  } else {
    DLOG << "Invalid pointer: " << address;
  }

  return paddr;
}

void *fpga_reg_malloc(size_t size) {
  void *ret = nullptr;
  ret = mmap64(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
               g_fpgainfo.fd_drv, FPGA_REG_PHY_ADDR);
  // PADDLE_MOBILE_ENFORCE(ret != (void *)-1, "Should not be -1");

  g_fpgainfo.fpga_addr2size_map.insert(std::make_pair(ret, size));

  return ret;
}

void *fpga_reg_free(void *ptr) {
  size_t size = 0;

  auto iter = g_fpgainfo.fpga_addr2size_map.find(ptr);
  if (iter != g_fpgainfo.fpga_addr2size_map.end()) {
    size = iter->second;
    g_fpgainfo.fpga_addr2size_map.erase(iter);
    munmap(ptr, size);
  } else {
    DLOG << "Invalid pointer" << ptr;
  }
}

static inline int do_ioctl(int64_t req, const void *arg) {
  return ioctl(g_fpgainfo.fd_mem, req, arg);
}

void *fpga_malloc_driver(size_t size) {
  void *ret = nullptr;
  uint64_t phy_addr = 0;
  int i = 0;
  struct MemoryVM2PHYArgs args;
  struct MemoryCacheArgs args_c;

  // memory_request(g_fpgainfo.memory_info, size, &phy_addr);

  ret = mmap64(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
               g_fpgainfo.fd_mem, FPGA_MEM_PHY_ADDR);
  PADDLE_MOBILE_ENFORCE(ret != (void *)-1, "Should not be -1");

  args.pVM = reinterpret_cast<void *>(ret);
  args.pPHY = reinterpret_cast<void *>(0);
  do_ioctl(IOCTL_MEMORY_VM2PHY, &args);
  phy_addr = (uint64_t)args.pPHY;

  g_fpgainfo.fpga_vaddr2paddr_map.insert(std::make_pair(ret, phy_addr));
  g_fpgainfo.fpga_addr2size_map.insert(std::make_pair(ret, size));

  return ret;
}

void fpga_free_driver(void *ptr) {
  size_t size = 0;
  uint32_t pos = 0;
  uint64_t p_addr = 0;

  auto iter = g_fpgainfo.fpga_addr2size_map.find(ptr);
  if (iter != g_fpgainfo.fpga_addr2size_map.end()) {
    size = iter->second;
    g_fpgainfo.fpga_addr2size_map.erase(iter);
    munmap(ptr, size);

    // p_addr = vaddr_to_paddr_driver(ptr);
    // pos = (p_addr - g_fpgainfo.memory_info->mem_start) / FPGA_PAGE_SIZE;

    auto iter = g_fpgainfo.fpga_vaddr2paddr_map.find(ptr);
    if (iter != g_fpgainfo.fpga_vaddr2paddr_map.end()) {
      g_fpgainfo.fpga_vaddr2paddr_map.erase(iter);
    }
  } else {
    DLOG << "Invalid pointer" << ptr;
  }
}

int fpga_flush_driver(void *address, size_t size) {
  struct MemoryCacheArgs args;
  uint64_t p_addr;

  p_addr = vaddr_to_paddr_driver(address);

  args.offset = (void *)(p_addr - FPGA_MEM_PHY_ADDR);  // NOLINT
  args.size = size;

  return do_ioctl(IOCTL_MEMCACHE_FLUSH, &args);
}

int fpga_invalidate_driver(void *address, size_t size) {
  struct MemoryCacheArgs args;
  uint64_t p_addr;

  p_addr = vaddr_to_paddr_driver(address);

  args.offset = (void *)(p_addr - FPGA_MEM_PHY_ADDR);  // NOLINT
  args.size = size;

  return do_ioctl(IOCTL_MEMCACHE_INVAL, &args);
}

void fpga_copy_driver(void *dest, const void *src, size_t num) {
  uint64_t i;
  for (i = 0; i < num; i++) {
    *((int8_t *)dest + i) = *((int8_t *)src + i);  // NOLINT
  }

  return;
}

int open_device_driver() {
  g_fpgainfo.FpgaRegPhyAddr = FPGA_REG_PHY_ADDR;
  g_fpgainfo.FpgaMemPhyAddr = FPGA_MEM_PHY_ADDR;
  g_fpgainfo.FpgaRegVirAddr = nullptr;
  g_fpgainfo.pe_data = nullptr;
  g_fpgainfo.drvdevice_path = "/dev/fpgadrv0";
  g_fpgainfo.memdevice_path = "/dev/fpgamem0";
  g_fpgainfo.fd_drv = -1;
  g_fpgainfo.fd_mem = -1;

  int ret = 0;
  ret = open_drvdevice();
  ret |= open_memdevice();

  g_fpgainfo.FpgaRegVirAddr =
      (uint64_t *)fpga_reg_malloc(FPGA_REG_SIZE);  // NOLINT
  // fpga_memory_add();

  pl_init();

  return ret;
}

int close_device_driver() {
  pl_destroy();
  fpga_reg_free(g_fpgainfo.FpgaRegVirAddr);
  memory_release(g_fpgainfo.memory_info);

  return 0;
}

}  // namespace driver
}  // namespace fpga
}  // namespace paddle_mobile
