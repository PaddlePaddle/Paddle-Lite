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
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "common/enforce.h"
#include "fpga/V2/driver/bitmap.h"
#include "fpga/V2/driver/driver.h"

namespace paddle_mobile {
namespace fpga {
struct FPGA_INFO g_fpgainfo;

int open_drvdevice() {
  if (g_fpgainfo.fd_drv == -1) {
    g_fpgainfo.fd_drv = open(g_fpgainfo.drvdevice_path, O_RDWR);
  }
  return g_fpgainfo.fd_drv;
}

int open_memdevice() {
  if (g_fpgainfo.fd_mem == -1) {
    g_fpgainfo.fd_mem = open(g_fpgainfo.memdevice_path, O_RDWR | O_DSYNC);
  }
  return g_fpgainfo.fd_mem;
}

void pl_reset() {
  // DLOG << "PL RESET";

  // reg_writeq(0x5a, REG_FPGA_RESET);
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
  int64_t timeout = time * CPU_FREQ / 1000000;

  for (i = 0; i < timeout; i++) {
    if (val == reg_readq(reg)) {
      break;
    }
  }

  if (i <= timeout) {
    return 0;
  } else {
    return -1;
  }
}

/*内存管理*/
int memory_request(struct fpga_memory *memory, size_t size, uint64_t *addr) {
  uint64_t _nr = DIV_ROUND_UP(size, FPGA_PAGE_SIZE);
  unsigned int nr = (unsigned int)_nr;
  int ret = 0;

  pthread_mutex_lock(&memory->mutex);

  unsigned int pos = (unsigned int)fpga_bitmap::bitmap_find_next_zero_area(
      memory->bitmap, memory->page_num, 0, nr, 0);
  if (pos <= memory->page_num) {
    uint64_t address_ofset =
        memory->mem_start + ((uint64_t)pos) * FPGA_PAGE_SIZE;
    fpga_bitmap::bitmap_set(memory->bitmap, pos, nr);
    memory->nr[pos] = nr;

    *addr = address_ofset;
  } else {
    ret = -ENOMEM;
  }

  pthread_mutex_unlock(&memory->mutex);

  return ret;
}

void memory_release(struct fpga_memory *memory) {
  pthread_mutex_lock(&memory->mutex);
  fpga_bitmap::bitmap_clear(memory->bitmap, 0, memory->page_num);
  pthread_mutex_unlock(&memory->mutex);
}

int create_fpga_memory_inner(struct fpga_memory *memory, size_t memory_size) {
  int rc = 0;

  uint64_t *bitmap = nullptr;
  unsigned int *nr = nullptr;

  // 不允许多份memory创建，所以创建memory结构体不存在互斥
  // pthread_mutex_lock(&memory->mutex);
  memory->page_num = (unsigned int)(memory_size / FPGA_PAGE_SIZE);
  memory->page_num_long = DIV_ROUND_UP(memory->page_num, BITS_PER_LONG);

  bitmap =
      (uint64_t *)malloc(sizeof(int64_t) * memory->page_num_long);  // NOLINT
  if (!bitmap) {
    rc = -EFAULT;
    return rc;
  }
  memory->bitmap = bitmap;

  nr = (unsigned int *)calloc(memory->page_num, sizeof(unsigned int));
  if (!nr) {
    rc = -EFAULT;
    free(bitmap);
    return rc;
  }
  memory->nr = nr;

  memory->mem_start = FPGA_MEM_PHY_ADDR;
  memory->mem_end = FPGA_MEM_SIZE;
  // pthread_mutex_unlock(memory->mutex);

  return rc;
}

int create_fpga_memory(struct fpga_memory **memory_info) {
  int rc = 0;

  *memory_info = (struct fpga_memory *)malloc(sizeof(struct fpga_memory));
  if (*memory_info == NULL) {
    rc = -EFAULT;
    return rc;
  }
  pthread_mutex_init(&((*memory_info)->mutex), nullptr);

  rc = create_fpga_memory_inner(*memory_info, FPGA_MEM_SIZE);
  if (rc) {
    free(*memory_info);
  }

  return rc;
}

int init_fpga_memory(struct fpga_memory *memory) {
  int rc = 0;

  if (!memory) {
    rc = -EFAULT;
    return rc;
  }

  // spin_lock_init(&memory->spin);
  fpga_bitmap::bitmap_clear(memory->bitmap, 0, memory->page_num);
  fpga_bitmap::bitmap_set(memory->bitmap, 0, 1);  // NOTE reserve fpga page 0.

  return 0;
}

void destroy_fpga_memory(struct fpga_memory *memory) {
  if (memory) {
    free(memory->nr);
    free(memory->bitmap);
    free(memory);
  }
}

int fpga_memory_add() {
  int rc = 0;

  rc = create_fpga_memory(&g_fpgainfo.memory_info);
  if (rc) {
    return rc;
  }

  rc = init_fpga_memory(g_fpgainfo.memory_info);
  if (rc) {
    destroy_fpga_memory(g_fpgainfo.memory_info);
    return rc;
  }

  return 0;
}

uint64_t vaddr_to_paddr(void *address) {
  uint64_t paddr = 0;
  auto iter = g_fpgainfo.fpga_vaddr2paddr_map.find(address);
  if (iter != g_fpgainfo.fpga_vaddr2paddr_map.end()) {
    paddr = iter->second;
  } else {
    DLOG << "Invalid pointer";
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

void *fpga_malloc_driver(size_t size) {
  void *ret = nullptr;
  uint64_t phy_addr = 0;

  memory_request(g_fpgainfo.memory_info, size, &phy_addr);

  ret = mmap64(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
               g_fpgainfo.fd_mem, phy_addr);
  PADDLE_MOBILE_ENFORCE(ret != (void *)-1, "Should not be -1");

  g_fpgainfo.fpga_vaddr2paddr_map.insert(std::make_pair(ret, phy_addr));
  g_fpgainfo.fpga_addr2size_map.insert(std::make_pair(ret, size));

  return ret;
}

void fpga_free_driver(void *ptr) {
  size_t size = 0;

  auto iter = g_fpgainfo.fpga_addr2size_map.find(ptr);
  if (iter != g_fpgainfo.fpga_addr2size_map.end()) {
    size = iter->second;
    g_fpgainfo.fpga_addr2size_map.erase(iter);
    munmap(ptr, size);
  } else {
    DLOG << "Invalid pointer";
  }
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
  fpga_memory_add();

  pl_init();

  return ret;
}

int close_device_driver() {
  pl_destroy();
  fpga_free_driver(g_fpgainfo.FpgaRegVirAddr);
  memory_release(g_fpgainfo.memory_info);
  destroy_fpga_memory(g_fpgainfo.memory_info);

  return 0;
}

}  // namespace fpga
}  // namespace paddle_mobile
