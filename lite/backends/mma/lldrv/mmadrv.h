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

#ifndef _LLDRV_MMA_H_
#define _LLDRV_MMA_H_

#pragma once

#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <limits>

namespace paddle {
namespace lite {
namespace mma {

// Activation type
enum mma_act_e {
  ACT_NONE = 0,
  ACT_RELU = 1,
};

// Device information
struct mma_info_s {
  uint32_t ver;  // Version, 00.00.0000
};

struct mma_reset_s {
  uint32_t val;  // reset command, N/A
};

// Memory copy
struct mma_mcopy_s {
  void* src;    // source address
  void* dst;    // destination adddress
  size_t size;  // size in bytes
};

// Memory block
struct mma_memblk_s {
  void* addr;   // base address
  size_t size;  // size in bytes
};

// Kernel
struct mma_kernel_s {
  uint32_t kw;  // width
  uint32_t kh;  // height
  uint32_t ws;  // width stride(s)
  uint32_t hs;  // height stride(s)
};

// Input parameters, nchw
struct mma_input_s {
  uint32_t in;  // nbr of batch {1}
  uint32_t ic;  // nbr of channels {1}
  uint32_t iw;  // width
  uint32_t ih;  // height
  uint32_t pl;  // padding x in bytes {0}
  uint32_t pr;  // padding x in bytes {0}
  uint32_t pt;  // padding y in bytes {0}
  uint32_t pb;  // padding y in bytes {0}
  uint32_t dx;  // dilation for x {1}
  uint32_t dy;  // dilation for y {1}
};

// Output parameters, nchw
struct mma_output_s {
  uint32_t on;  // nbr of batch {1}
  uint32_t oc;  // nbr of channels {1}
  uint32_t ow;  // width
  uint32_t oh;  // height
};

// Basic convolution
struct mma_conv_s {
  uint32_t at;            // activation type {0}, None=0, RELU=1
  uint32_t ng;            // nbr of groups {1}
  int8_t* ia;             // input address, INT8[N,Ci,Hi,Wi]
  int8_t* ka;             // kernel address, INT32[Co,Ci,Hk,Wk]
  int32_t* ba;            // bias address, INT32[Co,1]
  int32_t* oa;            // output address, INT32[N,Co,Ho,Wo]
  struct mma_input_s i;   // input
  struct mma_kernel_s k;  // kernel
  struct mma_output_s o;  // output
};

// Pooling convolution
struct mma_pool_s {
  uint32_t gp : 1;         // global pooling {0}
  uint32_t pm : 1;         // pooling mode {0}, Max=0, AVG=1
  uint32_t cm : 1;         // ceil mode {0}, ceil=0, floor=1
  uint32_t ex : 1;         // exclusive {1}, if ignore padding in avg pooling
  uint32_t reserved : 28;  // reserved {0}
  int32_t* ia;             // input address, INT32[N,Ci,Hi,Wi]
  int32_t* oa;             // output address, INT32[N,Ci,Ho,Wo]
  struct mma_input_s i;    // input
  struct mma_kernel_s k;   // kernel
  struct mma_output_s o;   // output
};

// Full connection
struct mma_fcon_s {
  uint32_t at;  // activation type {0}, None=0, RELU=1
  int8_t* ia;   // input address, INT8[M,K]
  int8_t* ka;   // kernel address, INT8[K,N]
  int32_t* ba;  // bias address, INT32[M,N]
  int32_t* oa;  // output address, INT32[M,N] = ia[M,K] * wa[K,N] + ba[M,N]
  int m, n, k;  // dims
};

// Regisger access
struct mma_creg_s {
  uint32_t addr;
  uint32_t data;
};

#define MMA_MAGIC_ID (('A' + 'L' + 'T' + 'R') / 4)

/* Ioctls */
#define MMA_IOCTL_MAKE(cmd) (_IO(MMA_MAGIC_ID, cmd))
#define MMA_IOCTL_GET(cmd) (_IOC_NR(cmd))
#define MMA_IOCTL_VALID(cmd) ((_IOC_TYPE(cmd) == MMA_MAGIC_ID) ? 1 : 0)

#define MMA_CMD_INFO 0x00   // struct mma_info_s
#define MMA_CMD_RESET 0x01  // struct mma_reset_s

#define MMA_CMD_MCOPY 0x10  // struct mma_mcopy_s
#define MMA_CMD_INVAL 0x11  // struct mma_cache_s
#define MMA_CMD_FLUSH 0x12  // struct mma_cache_s

#define MMA_CMD_CONV 0x20  // struct mma_conv_s
#define MMA_CMD_POOL 0x21  // struct mma_pool_s
#define MMA_CMD_FCON 0x22  // struct mma_fcon_s

#define MMA_CMD_REGRD 0xC0  // struct mma_register_s
#define MMA_CMD_REGWR 0xC1  // struct mma_register_s

//---------------------------------------------------------------------------

// device open/close
int mma_open();
void mma_close();

void mma_reset(struct mma_reset_s* args);

// memory management
void* mma_malloc(size_t size);
void mma_free(void* ptr);

void* mma_mbias(size_t size);
void* mma_mscale(size_t size);
void* mma_minput(size_t size);
void* mma_mkernel(size_t size);
void* mma_moutput(size_t size);

void mma_copy(void* dst, void* src, int size);
int mma_flush(void* addr, size_t size);
int mma_invalidate(void* addr, size_t size);

// device information
int mma_info(struct mma_info_s* args);

// convolution process
int mma_conv(struct mma_conv_s* args);
int mma_pooling(struct mma_pool_s* args);
int mma_fullconnect(struct mma_fcon_s* args);

}  // namespace mma
}  // namespace lite
}  // namespace paddle

#endif  // _LLDRV_MMA_H_
