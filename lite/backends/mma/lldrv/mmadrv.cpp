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

#include "lite/backends/mma/lldrv/mmadrv.h"

namespace paddle {
namespace lite {
namespace mma {

// FD of mma
static int mma_fd = -1;

// Memory blocks
static struct mma_memblk_s mb, ms, mi, mk, mo;

//---------------------------------------------------------------------------

int mma_open() {
  if (mma_fd < 0) {
    mma_fd = open("/dev/mmadrv0", O_RDWR);
    if (mma_fd < 0) {
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

void mma_close() {
  if (mma_fd < 0) return;

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
  close(mma_fd);
  mma_fd = -1;
}

//---------------------------------------------------------------------------

// memory management;
void* mma_malloc(size_t size) { return malloc(size); }

void mma_free(void* ptr) { free(ptr); }

void* mma_mbias(size_t size) {
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

void* mma_mscale(size_t size) {
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

void* mma_minput(size_t size) {
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

void* mma_mkernel(size_t size) {
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

void* mma_moutput(size_t size) {
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

void mma_copy(void* dst, void* src, int size) { memcpy(dst, src, size); }

//---------------------------------------------------------------------------

int mma_info(struct mma_info_s* args) {
  int cmd = MMA_IOCTL_MAKE(MMA_CMD_INFO);

  if (mma_open()) return -1;

  return ioctl(mma_fd, cmd, args);
}

int mma_conv(struct mma_conv_s* args) {
  int cmd = MMA_IOCTL_MAKE(MMA_CMD_CONV);

  if (mma_open()) return -1;

  return ioctl(mma_fd, cmd, args);
}

int mma_pooling(struct mma_pool_s* args) {
  int cmd = MMA_IOCTL_MAKE(MMA_CMD_POOL);

  if (mma_open()) return -1;

  return ioctl(mma_fd, cmd, args);
}

int mma_fullconnect(struct mma_fcon_s* args) {
  int cmd = MMA_IOCTL_MAKE(MMA_CMD_FCON);

  if (mma_open()) return -1;

  return ioctl(mma_fd, cmd, args);
}

//---------------------------------------------------------------------------

}  // namespace mma
}  // namespace lite
}  // namespace paddle
