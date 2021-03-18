/* Copyright (c) 2020 AWCloud. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <utility>

#include "lite/backends/intel_fpga/lldrv/intelfpgadrv.h"

namespace paddle {
namespace lite {
namespace intel_fpga {

/// FD of intel_fpga
static int intel_fpga_fd = -1;

/// Memory blocks
static struct intel_fpga_memblk_s mb, ms, mi, mk, mo;

int intel_fpga_open() {
  if (intel_fpga_fd < 0) {
    intel_fpga_fd = open("/dev/intelfpgadrv0", O_RDWR);
    if (intel_fpga_fd < 0) {
      return -1;
    }
    memset(&mb, 0, sizeof(mb));
    memset(&ms, 0, sizeof(ms));
    memset(&mi, 0, sizeof(mi));
    memset(&mk, 0, sizeof(mk));
    memset(&mo, 0, sizeof(mo));
  }

  return 0;
}

void intel_fpga_close() {
  if (intel_fpga_fd < 0) return;

  if (mb.addr) {
    free(mb.addr);
  }
  if (ms.addr) {
    free(ms.addr);
  }
  if (mi.addr) {
    free(mi.addr);
  }
  if (mk.addr) {
    free(mk.addr);
  }
  if (mo.addr) {
    free(mo.addr);
  }
  close(intel_fpga_fd);
  intel_fpga_fd = -1;
}

/// memory management;
void* intel_fpga_malloc(size_t size) { return malloc(size); }

void intel_fpga_free(void* ptr) { free(ptr); }

void* intel_fpga_mbias(size_t size) {
  if (mb.addr) {
    if (mb.size >= size) {
      return mb.addr;
    }
    free(mb.addr);
  }
  mb.addr = malloc(size);
  if (mb.addr) {
    mb.size = size;
  }
  return mb.addr;
}

void* intel_fpga_mscale(size_t size) {
  if (ms.addr) {
    if (ms.size >= size) {
      return ms.addr;
    }
    free(ms.addr);
  }
  ms.addr = malloc(size);
  if (ms.addr) {
    ms.size = size;
  }

  return ms.addr;
}

void* intel_fpga_minput(size_t size) {
  if (mi.addr) {
    if (mi.size >= size) {
      return mi.addr;
    }
    free(mi.addr);
  }
  mi.addr = malloc(size);
  if (mi.addr) {
    mi.size = size;
  }

  return mi.addr;
}

void* intel_fpga_mkernel(size_t size) {
  if (mk.addr) {
    if (mk.size >= size) {
      return mk.addr;
    }
    free(mk.addr);
  }
  mk.addr = malloc(size);
  if (mk.addr) {
    mk.size = size;
  }

  return mk.addr;
}

void* intel_fpga_moutput(size_t size) {
  if (mo.addr) {
    if (mo.size >= size) {
      return mo.addr;
    }
    free(mo.addr);
  }
  mo.addr = malloc(size);
  if (mo.addr) {
    mo.size = size;
  }

  return mo.addr;
}

void intel_fpga_copy(void* dst, void* src, int size) { memcpy(dst, src, size); }

int intel_fpga_info(struct intel_fpga_info_s* args) {
  int cmd = INTEL_FPGA_IOCTL_MAKE(INTEL_FPGA_CMD_INFO);

  if (intel_fpga_open()) return -1;

  return ioctl(intel_fpga_fd, cmd, args);
}

int intel_fpga_conv(struct intel_fpga_conv_s* args) {
  int cmd = INTEL_FPGA_IOCTL_MAKE(INTEL_FPGA_CMD_CONV);

  if (intel_fpga_open()) return -1;

  return ioctl(intel_fpga_fd, cmd, args);
}

int intel_fpga_pooling(struct intel_fpga_pool_s* args) {
  int cmd = INTEL_FPGA_IOCTL_MAKE(INTEL_FPGA_CMD_POOL);

  if (intel_fpga_open()) return -1;

  return ioctl(intel_fpga_fd, cmd, args);
}

int intel_fpga_fullconnect(struct intel_fpga_fcon_s* args) {
  int cmd = INTEL_FPGA_IOCTL_MAKE(INTEL_FPGA_CMD_FCON);

  if (intel_fpga_open()) return -1;

  return ioctl(intel_fpga_fd, cmd, args);
}

}  // namespace intel_fpga
}  // namespace lite
}  // namespace paddle
