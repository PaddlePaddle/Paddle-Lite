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

#pragma once

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <map>

#include "common/log.h"

namespace paddle_mobile {
namespace fpga {
namespace driver {

#define DIV_ROUND_UP(n, d) (((n) + (d)-1) / (d))

#define FPGA_REG_PHY_ADDR 0x80000000
#define FPGA_REG_SIZE 0x1000
#define FPGA_MEM_PHY_ADDR 0x20000000
#define FPGA_MEM_SIZE 0x20000000

#define FPGA_PAGE_SIZE (16UL * 1024UL)

// PE related macros
const int MAX_NUM_PES = 6;
const size_t MAX_TYPE_NAME_LENTH = 8;

const int PE_IDX_CONV = 0;
const int PE_IDX_POOLING = 1;
const int PE_IDX_EW = 2;
const int PE_IDX_BYPASS = 3;

enum pe_status { IDLE = 0, BUSY = 1, ERROR = 2 };

struct MemoryCacheArgs {
  void *offset;
  size_t size;
};

struct MemoryVM2PHYArgs {
  void *pVM;
  void *pPHY;
};

#define IOCTL_FPGA_MAGIC 'F'
#define IOCTL_MEMCACHE_INVAL _IOW(IOCTL_FPGA_MAGIC, 12, struct MemoryCacheArgs)
#define IOCTL_MEMCACHE_FLUSH _IOW(IOCTL_FPGA_MAGIC, 13, struct MemoryCacheArgs)
#define IOCTL_MEMORY_VM2PHY _IOWR(IOCTL_FPGA_MAGIC, 15, struct MemoryVM2PHYArgs)

struct fpga_pe {
  char type_name[MAX_TYPE_NAME_LENTH + 1];
  struct pe_data_s *outer;
  pe_status status;
  uint64_t interrupt_cnt;
};

struct pe_data_s {
  pthread_mutex_t mutex;
  struct fpga_pe pe_conv;
  struct fpga_pe pe_pooling;
  struct fpga_pe pe_ew;
  struct fpga_pe pe_bypass;

  struct fpga_pe *pes[MAX_NUM_PES];
  int pe_num;
};

struct fpga_memory {
  pthread_mutex_t mutex;
  uint64_t *bitmap;
  unsigned int *nr;
  unsigned int page_num;
  unsigned int page_num_long;
  uint64_t mem_start;
  uint64_t mem_end;
};

struct FPGA_INFO {
  uint64_t FpgaRegPhyAddr;
  uint64_t FpgaMemPhyAddr;
  pthread_t poll_pid;
  void *FpgaRegVirAddr;
  struct pe_data_s *pe_data;

  std::map<void *, size_t> fpga_addr2size_map;
  std::map<void *, uint64_t> fpga_vaddr2paddr_map;
  const char *drvdevice_path;
  const char *memdevice_path;
  struct fpga_memory *memory_info;
  int fd_drv;
  int fd_mem;
};

extern struct FPGA_INFO g_fpgainfo;

inline uint64_t reg_readq(uint32_t offset) {
  uint64_t value =
      *(volatile uint64_t *)((uint8_t *)g_fpgainfo.FpgaRegVirAddr +  // NOLINT
                             offset);                                // NOLINT
  return value;
}

inline void reg_writeq(uint64_t value, uint32_t offset) {
  *(volatile uint64_t *)((uint8_t *)g_fpgainfo.FpgaRegVirAddr +  // NOLINT
                         offset) = value;
}

int open_device_driver();

int close_device_driver();

void *fpga_malloc_driver(size_t size);

void fpga_free_driver(void *ptr);

int fpga_flush_driver(void *address, size_t size);

int fpga_invalidate_driver(void *address, size_t size);

uint64_t vaddr_to_paddr_driver(void *address);

int fpga_regpoll(uint64_t reg, uint64_t val, int time);

}  // namespace driver
}  // namespace fpga
}  // namespace paddle_mobile
