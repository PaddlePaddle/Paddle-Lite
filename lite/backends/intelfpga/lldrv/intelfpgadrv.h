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

#ifndef _LLDRV_INTELFPGA_H_
#define _LLDRV_INTELFPGA_H_

#pragma once

#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <limits>

namespace paddle {
namespace lite {
namespace intelfpga {

// Activation type
enum intelfpga_act_e {
  ACT_NONE = 0,
  ACT_RELU = 1,
};

// Device information
struct intelfpga_info_s {
  uint32_t ver;  // Version, 00.00.0000
};

struct intelfpga_reset_s {
  uint32_t val;  // reset command, N/A
};

// Memory copy
struct intelfpga_mcopy_s {
  void* src;    // source address
  void* dst;    // destination adddress
  size_t size;  // size in bytes
};

// Memory block
struct intelfpga_memblk_s {
  void* addr;   // base address
  size_t size;  // size in bytes
};

// Kernel
struct intelfpga_kernel_s {
  uint32_t kw;  // width
  uint32_t kh;  // height
  uint32_t ws;  // width stride(s)
  uint32_t hs;  // height stride(s)
};

// Input parameters, nchw
struct intelfpga_input_s {
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
struct intelfpga_output_s {
  uint32_t on;  // nbr of batch {1}
  uint32_t oc;  // nbr of channels {1}
  uint32_t ow;  // width
  uint32_t oh;  // height
};

// Basic convolution
struct intelfpga_conv_s {
  uint32_t at;                  // activation type {0}, None=0, RELU=1
  uint32_t ng;                  // nbr of groups {1}
  int8_t* ia;                   // input address, INT8[N,Ci,Hi,Wi]
  int8_t* ka;                   // kernel address, INT32[Co,Ci,Hk,Wk]
  int32_t* ba;                  // bias address, INT32[Co,1]
  int32_t* oa;                  // output address, INT32[N,Co,Ho,Wo]
  struct intelfpga_input_s i;   // input
  struct intelfpga_kernel_s k;  // kernel
  struct intelfpga_output_s o;  // output
};

// Pooling convolution
struct intelfpga_pool_s {
  uint32_t gp : 1;         // global pooling {0}
  uint32_t pm : 1;         // pooling mode {0}, Max=0, AVG=1
  uint32_t cm : 1;         // ceil mode {0}, ceil=0, floor=1
  uint32_t ex : 1;         // exclusive {1}, if ignore padding in avg pooling
  uint32_t reserved : 28;  // reserved {0}
  int32_t* ia;             // input address, INT32[N,Ci,Hi,Wi]
  int32_t* oa;             // output address, INT32[N,Ci,Ho,Wo]
  struct intelfpga_input_s i;   // input
  struct intelfpga_kernel_s k;  // kernel
  struct intelfpga_output_s o;  // output
};

// Full connection
struct intelfpga_fcon_s {
  uint32_t at;  // activation type {0}, None=0, RELU=1
  int8_t* ia;   // input address, INT8[M,K]
  int8_t* ka;   // kernel address, INT8[K,N]
  int32_t* ba;  // bias address, INT32[M,N]
  int32_t* oa;  // output address, INT32[M,N] = ia[M,K] * wa[K,N] + ba[M,N]
  int m, n, k;  // dims
};

// Regisger access
struct intelfpga_creg_s {
  uint32_t addr;
  uint32_t data;
};

#define INTELFPGA_MAGIC_ID (('A' + 'L' + 'T' + 'R') / 4)

/* Ioctls */
#define INTELFPGA_IOCTL_MAKE(cmd) (_IO(INTELFPGA_MAGIC_ID, cmd))
#define INTELFPGA_IOCTL_GET(cmd) (_IOC_NR(cmd))
#define INTELFPGA_IOCTL_VALID(cmd) \
  ((_IOC_TYPE(cmd) == INTELFPGA_MAGIC_ID) ? 1 : 0)

#define INTELFPGA_CMD_INFO 0x00   // struct intelfpga_info_s
#define INTELFPGA_CMD_RESET 0x01  // struct intelfpga_reset_s

#define INTELFPGA_CMD_MCOPY 0x10  // struct intelfpga_mcopy_s
#define INTELFPGA_CMD_INVAL 0x11  // struct intelfpga_cache_s
#define INTELFPGA_CMD_FLUSH 0x12  // struct intelfpga_cache_s

#define INTELFPGA_CMD_CONV 0x20  // struct intelfpga_conv_s
#define INTELFPGA_CMD_POOL 0x21  // struct intelfpga_pool_s
#define INTELFPGA_CMD_FCON 0x22  // struct intelfpga_fcon_s

#define INTELFPGA_CMD_REGRD 0xC0  // struct intelfpga_register_s
#define INTELFPGA_CMD_REGWR 0xC1  // struct intelfpga_register_s

//---------------------------------------------------------------------------

// device open/close
int intelfpga_open();
void intelfpga_close();

void intelfpga_reset(struct intelfpga_reset_s* args);

// memory management
void* intelfpga_malloc(size_t size);
void intelfpga_free(void* ptr);

void* intelfpga_mbias(size_t size);
void* intelfpga_mscale(size_t size);
void* intelfpga_minput(size_t size);
void* intelfpga_mkernel(size_t size);
void* intelfpga_moutput(size_t size);

void intelfpga_copy(void* dst, void* src, int size);
int intelfpga_flush(void* addr, size_t size);
int intelfpga_invalidate(void* addr, size_t size);

// device information
int intelfpga_info(struct intelfpga_info_s* args);

// convolution process
int intelfpga_conv(struct intelfpga_conv_s* args);
int intelfpga_pooling(struct intelfpga_pool_s* args);
int intelfpga_fullconnect(struct intelfpga_fcon_s* args);

}  // namespace intelfpga
}  // namespace lite
}  // namespace paddle

#endif  // _LLDRV_INTELFPGA_H_
