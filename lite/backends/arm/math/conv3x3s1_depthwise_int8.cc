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

#include <arm_neon.h>
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))
// clang-format off
#ifdef __aarch64__
#define INIT_S1                           \
  "PRFM PLDL1KEEP, [%[din_ptr0]] \n"      \
  "PRFM PLDL1KEEP, [%[din_ptr1]] \n"      \
  "PRFM PLDL1KEEP, [%[din_ptr2]] \n"      \
  "PRFM PLDL1KEEP, [%[din_ptr3]] \n"      \
  "movi   v21.4s, #0x0\n"                 \
  "ld1    {v0.8b, v1.8b}, [%[din_ptr0]]\n"\
  "ld1    {v2.8b, v3.8b}, [%[din_ptr1]]\n"\
  "movi   v12.4s, #0x0\n"                 \
  "movi   v13.4s, #0x0\n"                 \
  "movi   v14.4s, #0x0\n"                 \
  "movi   v15.4s, #0x0\n"

#define LEFT_COMPUTE_S1                     \
  /* r0 */                                  \
  "ext    v4.8b,  v21.8b, v0.8b, #7   \n"   \
  "ext    v6.8b,  v21.8b, v2.8b, #7   \n"   \
  "smull  v16.8h, v0.8b,  %[v1].8b    \n"   \
  "smull  v18.8h, v2.8b,  %[v1].8b    \n"   \
  "ext    v5.8b,  v0.8b,  v1.8b, #1   \n"   \
  "ext    v7.8b,  v2.8b,  v3.8b, #1   \n"   \
  "add   %[din_ptr0], %[din_ptr0], #7 \n"   \
  "add   %[din_ptr1], %[din_ptr1], #7 \n"   \
  "smlal v16.8h,  v4.8b,  %[v0].8b    \n"   \
  "smlal v18.8h,  v6.8b,  %[v0].8b    \n"   \
  "ld1    {v0.8b, v1.8b}, [%[din_ptr2]]\n"  \
  "smull  v17.8h, v5.8b,  %[v2].8b    \n"   \
  "smull  v19.8h, v7.8b,  %[v2].8b    \n"   \
  "saddw  v12.4s, v12.4s, v16.4h       \n"  \
  "saddw2 v13.4s, v13.4s, v16.8h       \n"  \
  "saddw  v14.4s, v14.4s, v18.4h       \n"  \
  "saddw2 v15.4s, v15.4s, v18.8h       \n"  \
  /* r1 */                                  \
  "ext    v4.8b,  v21.8b, v0.8b, #7   \n"   \
  "ext    v5.8b,  v0.8b,  v1.8b, #1   \n"   \
  "smlal v17.8h,  v2.8b,  %[v4].8b    \n"   \
  "smlal v19.8h,  v0.8b,  %[v4].8b    \n"   \
  "smull  v16.8h, v6.8b,  %[v3].8b    \n"   \
  "smull  v18.8h, v4.8b,  %[v3].8b    \n"   \
  "ld1    {v2.8b, v3.8b}, [%[din_ptr3]]\n"  \
  "saddw  v12.4s, v12.4s, v17.4h       \n"  \
  "saddw2 v13.4s, v13.4s, v17.8h       \n"  \
  "saddw  v14.4s, v14.4s, v19.4h       \n"  \
  "saddw2 v15.4s, v15.4s, v19.8h       \n"  \
  "smlal  v16.8h, v7.8b,  %[v5].8b    \n"   \
  "smlal  v18.8h, v5.8b,  %[v5].8b    \n"   \
  /* r2 */                                  \
  "ext    v6.8b,  v21.8b, v2.8b, #7   \n"   \
  "ext    v7.8b,  v2.8b,  v3.8b, #1   \n"   \
  "smull  v17.8h, v0.8b,  %[v7].8b    \n"   \
  "smull  v19.8h, v2.8b,  %[v7].8b    \n"   \
  "saddw  v12.4s, v12.4s, v16.4h      \n"   \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"   \
  "saddw  v14.4s, v14.4s, v18.4h      \n"   \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"   \
  "smull  v16.8h, v4.8b,  %[v6].8b    \n"   \
  "smull  v18.8h, v6.8b,  %[v6].8b    \n"   \
  "smlal  v17.8h, v5.8b,  %[v8].8b    \n"   \
  "smlal  v19.8h, v7.8b,  %[v8].8b    \n"   \
  "dup    v8.4s,  %w[bias]            \n"   \
  "dup    v9.4s,  %w[bias]            \n"   \
  "dup    v10.4s, %w[bias]            \n"   \
  "dup    v11.4s, %w[bias]            \n"   \
  "saddw  v12.4s, v12.4s, v16.4h      \n"   \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"   \
  "saddw  v14.4s, v14.4s, v18.4h      \n"   \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"   \
  "add   %[din_ptr2], %[din_ptr2], #7 \n"   \
  "add   %[din_ptr3], %[din_ptr3], #7 \n"   \
  "saddw  v12.4s, v12.4s, v17.4h      \n"   \
  "saddw2 v13.4s, v13.4s, v17.8h      \n"   \
  "saddw  v14.4s, v14.4s, v19.4h      \n"   \
  "saddw2 v15.4s, v15.4s, v19.8h      \n"   \
  /* int32 -> fp32 */                       \
  "scvtf  v12.4s, v12.4s              \n"   \
  "scvtf  v13.4s, v13.4s              \n"   \
  "scvtf  v14.4s, v14.4s              \n"   \
  "scvtf  v15.4s, v15.4s              \n"   \
  "fmla   v8.4s,  v12.4s, %[vscale].4s\n"   \
  "fmla   v9.4s,  v13.4s, %[vscale].4s\n"   \
  "fmla   v10.4s, v14.4s, %[vscale].4s\n"   \
  "fmla   v11.4s, v15.4s, %[vscale].4s\n"

#define RELU                                \
  "fmax  v8.4s,  v8.4s,  v21.4s        \n"  \
  "fmax  v9.4s,  v9.4s,  v21.4s        \n"  \
  "fmax  v10.4s, v10.4s, v21.4s        \n"  \
  "fmax  v11.4s, v11.4s, v21.4s        \n"

#define RELU6                        \
  "ldr   q4, [%[vmax], #16]     \n"  \
  "fmax  v8.4s,  v8.4s,  v21.4s \n"  \
  "fmax  v9.4s,  v9.4s,  v21.4s \n"  \
  "fmax  v10.4s, v10.4s, v21.4s \n"  \
  "fmax  v11.4s, v11.4s, v21.4s \n"  \
  "fmin  v8.4s,  v8.4s,  v4.4s  \n"  \
  "fmin  v9.4s,  v9.4s,  v4.4s  \n"  \
  "fmin  v10.4s, v10.4s, v4.4s  \n"  \
  "fmin  v11.4s, v11.4s, v4.4s  \n"

#define RESULT_INT8_MAX                    \
  "ld1   {v12.4s}, [%[vmax]]           \n" \
  "fcmge v4.4s,   v8.4s,  v12.4s       \n" \
  "fcmge v5.4s,   v9.4s,  v12.4s       \n" \
  "fcmge v6.4s,   v10.4s, v12.4s       \n" \
  "fcmge v7.4s,   v11.4s, v12.4s       \n" \
  "bif   v8.16b,  v12.16b, v4.16b      \n" \
  "bif   v9.16b,  v12.16b, v5.16b      \n" \
  "bif   v10.16b, v12.16b, v6.16b      \n" \
  "bif   v11.16b, v12.16b, v7.16b      \n"

#define RESULT_INT8                        \
  /* fp32 -> int32 */                      \
  "fcvtas  v8.4s,  v8.4s               \n" \
  "fcvtas  v9.4s,  v9.4s               \n" \
  "fcvtas  v10.4s, v10.4s              \n" \
  "fcvtas  v11.4s, v11.4s              \n" \
  /* int32 -> int16 */                     \
  "sqxtn   v12.4h,  v8.4s              \n" \
  "sqxtn2  v12.8h,  v9.4s              \n" \
  "sqxtn   v13.4h,  v10.4s             \n" \
  "sqxtn2  v13.8h,  v11.4s             \n" \
  /* int16 -> int8  */                     \
  "sqxtn   v8.8b,  v12.8h              \n" \
  "sqxtn   v9.8b,  v13.8h              \n"

#define LEFT_STORE_FLOAT                   \
  "stp    q8,  q9,  [%[dout_ptr0]], #32 \n" \
  "stp    q10, q11, [%[dout_ptr1]], #32 \n"

#define LEFT_STORE_INT8                       \
  "st1    {v8.8b}, [%[dout_ptr0]], #8     \n" \
  "st1    {v9.8b}, [%[dout_ptr1]], #8     \n"

#define MID_COMPUTE_S1                        \
  "cmp  %[cnt], #1                     \n"    \
  "ld1    {v0.8b, v1.8b}, [%[din_ptr0]]\n"    \
  "ld1    {v2.8b, v3.8b}, [%[din_ptr1]]\n"    \
  "movi   v12.4s, #0x0\n"                     \
  "movi   v13.4s, #0x0\n"                     \
  "movi   v14.4s, #0x0\n"                     \
  "movi   v15.4s, #0x0\n"                     \
  "blt 1f             \n"                     \
  "2:                 \n"                     \
  /* r0 */                                    \
  "ext    v4.8b,  v0.8b,  v1.8b, #1   \n"     \
  "ext    v6.8b,  v2.8b,  v3.8b, #1   \n"     \
  "smull  v16.8h, v0.8b,  %[v0].8b    \n"     \
  "smull  v18.8h, v2.8b,  %[v0].8b    \n"     \
  "ext    v5.8b,  v0.8b,  v1.8b, #2   \n"     \
  "ext    v7.8b,  v2.8b,  v3.8b, #2   \n"     \
  "smull  v17.8h, v4.8b,  %[v1].8b    \n"     \
  "smull  v19.8h, v6.8b,  %[v1].8b    \n"     \
  "ld1    {v0.8b, v1.8b}, [%[din_ptr2]]\n"    \
  "add    %[din_ptr0], %[din_ptr0], #8\n"     \
  "add    %[din_ptr1], %[din_ptr1], #8\n"     \
  "smlal  v16.8h, v5.8b,  %[v2].8b    \n"     \
  "smlal  v18.8h, v7.8b,  %[v2].8b    \n"     \
  /* r1 */                                    \
  "ext    v4.8b,  v0.8b,  v1.8b, #1   \n"     \
  "ext    v5.8b,  v0.8b,  v1.8b, #2   \n"     \
  "smlal  v17.8h, v2.8b,  %[v3].8b    \n"     \
  "smlal  v19.8h, v0.8b,  %[v3].8b    \n"     \
  "saddw  v12.4s, v12.4s, v16.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"     \
  "saddw  v14.4s, v14.4s, v18.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"     \
  "ld1    {v2.8b, v3.8b}, [%[din_ptr3]]\n"    \
  "smull  v16.8h, v6.8b,  %[v4].8b    \n"     \
  "smull  v18.8h, v4.8b,  %[v4].8b    \n"     \
  "saddw  v12.4s, v12.4s, v17.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v17.8h      \n"     \
  "saddw  v14.4s, v14.4s, v19.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v19.8h      \n"     \
  "add    %[din_ptr2], %[din_ptr2], #8\n"     \
  "add    %[din_ptr3], %[din_ptr3], #8\n"     \
  "smull  v17.8h, v7.8b,  %[v5].8b    \n"     \
  "smull  v19.8h, v5.8b,  %[v5].8b    \n"     \
  /* r2 */                                    \
  "ext    v6.8b,  v2.8b,  v3.8b, #1   \n"     \
  "ext    v7.8b,  v2.8b,  v3.8b, #2   \n"     \
  "smlal  v16.8h, v0.8b,  %[v6].8b    \n"     \
  "smlal  v18.8h, v2.8b,  %[v6].8b    \n"     \
  "ld1    {v0.8b, v1.8b}, [%[din_ptr0]]\n"    \
  "ld1    {v2.8b, v3.8b}, [%[din_ptr1]]\n"    \
  "smlal  v17.8h, v4.8b,  %[v7].8b    \n"     \
  "smlal  v19.8h, v6.8b,  %[v7].8b    \n"     \
  "saddw  v12.4s, v12.4s, v16.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"     \
  "saddw  v14.4s, v14.4s, v18.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"     \
  "smull  v16.8h, v5.8b,  %[v8].8b    \n"     \
  "smull  v18.8h, v7.8b,  %[v8].8b    \n"     \
  "saddw  v12.4s, v12.4s, v17.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v17.8h      \n"     \
  "saddw  v14.4s, v14.4s, v19.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v19.8h      \n"     \
  "dup    v8.4s,  %w[bias]            \n"     \
  "dup    v9.4s,  %w[bias]            \n"     \
  "dup    v10.4s, %w[bias]            \n"     \
  "dup    v11.4s, %w[bias]            \n"     \
  "saddw  v12.4s, v12.4s, v16.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"     \
  "saddw  v14.4s, v14.4s, v18.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"     \
  /* int32 -> fp32 */                         \
  "scvtf  v12.4s, v12.4s              \n"     \
  "scvtf  v13.4s, v13.4s              \n"     \
  "scvtf  v14.4s, v14.4s              \n"     \
  "scvtf  v15.4s, v15.4s              \n"     \
  "subs   %w[cnt], %w[cnt], #1        \n"     \
  "fmla   v8.4s,  v12.4s, %[vscale].4s\n"     \
  "fmla   v9.4s,  v13.4s, %[vscale].4s\n"     \
  "fmla   v10.4s, v14.4s, %[vscale].4s\n"     \
  "fmla   v11.4s, v15.4s, %[vscale].4s\n"

#define MID_STORE_FLOAT                      \
  "movi   v12.4s, #0x0\n"                    \
  "movi   v13.4s, #0x0\n"                    \
  "movi   v14.4s, #0x0\n"                    \
  "movi   v15.4s, #0x0\n"                    \
  "stp    q8,  q9,  [%[dout_ptr0]], #32 \n"  \
  "stp    q10, q11, [%[dout_ptr1]], #32 \n"  \
  "bne  2b\n"

#define MID_STORE_INT8                       \
  "movi   v12.4s, #0x0\n"                    \
  "movi   v13.4s, #0x0\n"                    \
  "movi   v14.4s, #0x0\n"                    \
  "movi   v15.4s, #0x0\n"                    \
  "st1    {v8.8b}, [%[dout_ptr0]], #8     \n"\
  "st1    {v9.8b}, [%[dout_ptr1]], #8     \n"\
  "bne  2b\n"

#define RIGHT_COMPUTE_S1                    \
  "1:                                \n"    \
  "sub %[din_ptr0], %[din_ptr0], %[right_pad_num_in]\n"\
  "sub %[din_ptr1], %[din_ptr1], %[right_pad_num_in]\n"\
  "sub %[din_ptr2], %[din_ptr2], %[right_pad_num_in]\n"\
  "sub %[din_ptr3], %[din_ptr3], %[right_pad_num_in]\n"\
  "ld1    {v0.8b, v1.8b}, [%[din_ptr0]]\n"    \
  "ld1    {v2.8b, v3.8b}, [%[din_ptr1]]\n"    \
  "ld1    {v9.8b}, [%[vmask]]\n"       \
  "sub %[dout_ptr0], %[dout_ptr0], %[right_pad_num_out]\n"\
  "sub %[dout_ptr1], %[dout_ptr1], %[right_pad_num_out]\n"\
  "bif    v1.8b,  v21.8b, v9.8b       \n"     \
  "bif    v3.8b,  v21.8b, v9.8b       \n"     \
  /* r0 */                                    \
  "ext    v4.8b,  v0.8b,  v1.8b, #1   \n"     \
  "ext    v6.8b,  v2.8b,  v3.8b, #1   \n"     \
  "smull  v16.8h, v0.8b,  %[v0].8b    \n"     \
  "smull  v18.8h, v2.8b,  %[v0].8b    \n"     \
  "ext    v5.8b,  v0.8b,  v1.8b, #2   \n"     \
  "ext    v7.8b,  v2.8b,  v3.8b, #2   \n"     \
  "ld1    {v0.8b, v1.8b}, [%[din_ptr2]]\n"    \
  "smull  v17.8h, v4.8b,  %[v1].8b    \n"     \
  "smull  v19.8h, v6.8b,  %[v1].8b    \n"     \
  "smlal  v16.8h, v5.8b,  %[v2].8b    \n"     \
  "smlal  v18.8h, v7.8b,  %[v2].8b    \n"     \
  "bif    v1.8b,  v21.8b, v9.8b       \n"     \
  /* r1 */                                    \
  "ext    v4.8b,  v0.8b,  v1.8b, #1   \n"     \
  "ext    v5.8b,  v0.8b,  v1.8b, #2   \n"     \
  "smlal  v17.8h, v2.8b,  %[v3].8b    \n"     \
  "smlal  v19.8h, v0.8b,  %[v3].8b    \n"     \
  "saddw  v12.4s, v12.4s, v16.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"     \
  "saddw  v14.4s, v14.4s, v18.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"     \
  "ld1    {v2.8b, v3.8b}, [%[din_ptr3]]\n"    \
  "smull  v16.8h, v6.8b,  %[v4].8b    \n"     \
  "smull  v18.8h, v4.8b,  %[v4].8b    \n"     \
  "saddw  v12.4s, v12.4s, v17.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v17.8h      \n"     \
  "saddw  v14.4s, v14.4s, v19.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v19.8h      \n"     \
  "bif    v3.8b,  v21.8b, v9.8b       \n"     \
  "smull  v17.8h, v7.8b,  %[v5].8b    \n"     \
  "smull  v19.8h, v5.8b,  %[v5].8b    \n"     \
  /* r2 */                                    \
  "ext    v6.8b,  v2.8b,  v3.8b, #1   \n"     \
  "ext    v7.8b,  v2.8b,  v3.8b, #2   \n"     \
  "smlal  v16.8h, v0.8b,  %[v6].8b    \n"     \
  "smlal  v18.8h, v2.8b,  %[v6].8b    \n"     \
  "smlal  v17.8h, v4.8b,  %[v7].8b    \n"     \
  "smlal  v19.8h, v6.8b,  %[v7].8b    \n"     \
  "saddw  v12.4s, v12.4s, v16.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"     \
  "saddw  v14.4s, v14.4s, v18.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"     \
  "smull  v16.8h, v5.8b,  %[v8].8b    \n"     \
  "smull  v18.8h, v7.8b,  %[v8].8b    \n"     \
  "saddw  v12.4s, v12.4s, v17.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v17.8h      \n"     \
  "saddw  v14.4s, v14.4s, v19.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v19.8h      \n"     \
  "dup    v8.4s,  %w[bias]            \n"     \
  "dup    v9.4s,  %w[bias]            \n"     \
  "dup    v10.4s, %w[bias]            \n"     \
  "dup    v11.4s, %w[bias]            \n"     \
  "saddw  v12.4s, v12.4s, v16.4h      \n"     \
  "saddw2 v13.4s, v13.4s, v16.8h      \n"     \
  "saddw  v14.4s, v14.4s, v18.4h      \n"     \
  "saddw2 v15.4s, v15.4s, v18.8h      \n"     \
  /* int32 -> fp32 */                         \
  "scvtf  v12.4s, v12.4s              \n"     \
  "scvtf  v13.4s, v13.4s              \n"     \
  "scvtf  v14.4s, v14.4s              \n"     \
  "scvtf  v15.4s, v15.4s              \n"     \
  "fmla   v8.4s,  v12.4s, %[vscale].4s\n"     \
  "fmla   v9.4s,  v13.4s, %[vscale].4s\n"     \
  "fmla   v10.4s, v14.4s, %[vscale].4s\n"     \
  "fmla   v11.4s, v15.4s, %[vscale].4s\n"

#else
#define INIT_S1                           \
  "vld1.8    {d0-d1}, [%[wei_ptr]]  \n"   \
  "pld [%[din_ptr0]]                \n"   \
  "pld [%[din_ptr1]]                \n"   \
  "pld [%[din_ptr2]]                \n"   \
  "pld [%[din_ptr3]]                \n"   \
  "vdup.s8     d2, d0[0]            \n"   \
  "vdup.s8     d3, d0[1]            \n"   \
  "vdup.s8     d4, d0[2]            \n"   \
  "vld1.8 {d12-d13}, [%[din_ptr0]]  \n"   \
  "vmov.u32 d11, #0                 \n"   \
  "vmov.u32 q12, #0                 \n"   \
  "vld1.8 {d16-d17}, [%[din_ptr1]]  \n"   \
  "vmov.u32 q13, #0                 \n"   \
  "vmov.u32 q14, #0                 \n"   \
  "vmov.u32 q15, #0                 \n"

#define LEFT_COMPUTE_S1                   \
  /* r0 */                                \
  "vmull.s8 q10, d12, d3            \n"   \
  "vmull.s8 q11, d16, d3            \n"   \
  "vext.8   d14, d11, d12, #7       \n"   \
  "vext.8   d18, d11, d16, #7       \n"   \
  "vext.8   d15, d12, d13, #1       \n"   \
  "vext.8   d19, d16, d17, #1       \n"   \
  "vmlal.s8 q10, d14, d2            \n"   \
  "vmlal.s8 q11, d18, d2            \n"   \
  "vdup.s8     d5, d0[3]            \n"   \
  "vdup.s8     d6, d0[4]            \n"   \
  "vdup.s8     d7, d0[5]            \n"   \
  "vld1.8 {d12-d13}, [%[din_ptr2]]  \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8 q10, d15, d4            \n"   \
  "vmull.s8 q11, d19, d4            \n"   \
  "add      %[din_ptr0], #7         \n"   \
  /* r1 */                                \
  "vext.8   d14, d11, d12, #7       \n"   \
  "vext.8   d15, d12, d13, #1       \n"   \
  "vmlal.s8 q10, d16,  d6           \n"   \
  "vmlal.s8 q11, d12,  d6           \n"   \
  "add      %[din_ptr1], #7         \n"   \
  "vld1.8 {d16-d17}, [%[din_ptr3]]  \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8  q10, d18, d5           \n"   \
  "vmull.s8  q11, d14, d5           \n"   \
  "vdup.s8     d8, d0[6]            \n"   \
  "vdup.s8     d9, d0[7]            \n"   \
  "vdup.s8     d10, d1[0]           \n"   \
  "vmlal.s8  q10, d19, d7           \n"   \
  "vmlal.s8  q11, d15, d7           \n"   \
  /* r2 */                                \
  "vext.8   d18, d11, d16, #7       \n"   \
  "vext.8   d19, d16, d17, #1       \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8  q10, d12, d9           \n"   \
  "vmull.s8  q11, d16, d9           \n"   \
  "vmull.s8  q6,  d14, d8           \n"   \
  "vmull.s8  q8,  d18, d8           \n"   \
  "add      %[din_ptr2], #7         \n"   \
  "add      %[din_ptr3], #7         \n"   \
  "vmlal.s8  q10, d15, d10          \n"   \
  "vmlal.s8  q11, d19, d10         \n"    \
  "vaddw.s16 q12, q12, d12          \n"   \
  "vaddw.s16 q13, q13, d13          \n"   \
  "vaddw.s16 q14, q14, d16          \n"   \
  "vaddw.s16 q15, q15, d17          \n"   \
  "vld1.32   {d14-d15}, [%[vmax]]   \n"   \
  "vdup.32   q8, %[bias]            \n"   \
  "vdup.32   q9, %[bias]            \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vdup.32   q10, %[bias]           \n"   \
  "vdup.32   q11, %[bias]           \n"   \
  "vcvt.f32.s32   q12, q12          \n"   \
  "vcvt.f32.s32   q13, q13          \n"   \
  "vcvt.f32.s32   q14, q14          \n"   \
  "vcvt.f32.s32   q15, q15          \n"   \
  "vmov.u32 q6, #0                  \n"   \
  "vmla.f32 q8,   q12,  q7          \n"   \
  "vmla.f32 q9,   q13,  q7          \n"   \
  "vmla.f32 q10,  q14,  q7          \n"   \
  "vmla.f32 q11,  q15,  q7          \n"

#define RELU6                             \
  "vldr     d14, [%[vmax], #32]    \n"    \
  "vldr     d15, [%[vmax], #40]    \n"    \
  "vmax.f32 q8,  q8,  q6           \n"    \
  "vmax.f32 q9,  q9,  q6           \n"    \
  "vmax.f32 q10, q10, q6           \n"    \
  "vmax.f32 q11, q11, q6           \n"    \
  "vmin.f32 q8,  q8,  q7           \n"    \
  "vmin.f32 q9,  q9,  q7           \n"    \
  "vmin.f32 q10, q10, q7           \n"    \
  "vmin.f32 q11, q11, q7           \n"

#define RELU                              \
  "vmax.f32 q8,  q8,  q6           \n"    \
  "vmax.f32 q9,  q9,  q6           \n"    \
  "vmax.f32 q10, q10, q6           \n"    \
  "vmax.f32 q11, q11, q6           \n"

#define RESULT_INT8                        \
  "vmov.f32 q12, #-0.5                \n"  \
  "vmov.f32 q13, #0.5                 \n"  \
  "vcgt.f32 q14, q8,  q6              \n"  \
  "vcgt.f32 q15, q9,  q6              \n"  \
  "vldr     d14,  [%[vmax], #16]      \n"  \
  "vldr     d15,  [%[vmax], #24]      \n"  \
  "vbif.f32 q13, q12, q14             \n"  \
  "vcgt.f32 q14, q10, q6              \n"  \
  "vadd.f32 q8,  q8,  q13             \n"  \
  "vmov.f32 q13, #0.5                 \n"  \
  "vbif.f32 q13, q12, q15             \n"  \
  "vcgt.f32 q15, q11, q6              \n"  \
  "vadd.f32 q9, q9,  q13              \n"  \
  "vmov.f32 q13, #0.5                 \n"  \
  "vbif.f32 q13, q12, q14             \n"  \
  "vadd.f32 q10, q10, q13             \n"  \
  "vmov.f32 q13, #0.5                 \n"  \
  "vbif.f32 q13, q12, q15             \n"  \
  "vadd.f32 q11, q11, q13             \n"  \
  /* >= -127 */                            \
  "vcgt.f32 q12, q8,  q7              \n"  \
  "vcgt.f32 q13, q9,  q7              \n"  \
  "vcgt.f32 q14, q10, q7              \n"  \
  "vcgt.f32 q15, q11, q7              \n"  \
  "vbif.f32 q8,  q7,  q12             \n"  \
  "vbif.f32 q9,  q7,  q13             \n"  \
  "vbif.f32 q10, q7,  q14             \n"  \
  "vbif.f32 q11, q7,  q15             \n"  \
  /* f32 -> int32 */                       \
  "vcvt.s32.f32  q12, q8              \n"  \
  "vcvt.s32.f32  q13, q9              \n"  \
  "vcvt.s32.f32  q14, q10             \n"  \
  "vcvt.s32.f32  q15, q11             \n"  \
  /* int32 -> int16 */                     \
  "vqmovn.s32 d16, q12                \n"  \
  "vqmovn.s32 d17, q13                \n"  \
  "vqmovn.s32 d18, q14                \n"  \
  "vqmovn.s32 d19, q15                \n"  \
  /* int16 -> int8 */                      \
  "vqmovn.s16 d20, q8                 \n"  \
  "vqmovn.s16 d21, q9                 \n"

#define LEFT_STORE_FLOAT                  \
  "vst1.32 {d16-d19}, [%[dout_ptr0]]!\n"  \
  "vst1.32 {d20-d23}, [%[dout_ptr1]]!\n"

#define LEFT_STORE_INT8                   \
  "vst1.32 {d20}, [%[dout_ptr0]]!\n"      \
  "vst1.32 {d21}, [%[dout_ptr1]]!\n"

#define INIT_P0                           \
  "vld1.8    {d0-d1}, [%[wei_ptr]]  \n"   \
  "pld [%[din_ptr0]]                \n"   \
  "pld [%[din_ptr1]]                \n"   \
  "pld [%[din_ptr2]]                \n"   \
  "pld [%[din_ptr3]]                \n"   \
  "vdup.s8     d2,  d0[0]           \n"   \
  "vdup.s8     d3,  d0[1]           \n"   \
  "vdup.s8     d4,  d0[2]           \n"   \
  "vdup.s8     d5,  d0[3]           \n"   \
  "vdup.s8     d6,  d0[4]           \n"   \
  "vdup.s8     d7,  d0[5]           \n"   \
  "vdup.s8     d8,  d0[6]           \n"   \
  "vdup.s8     d9,  d0[7]           \n"   \
  "vdup.s8     d10, d1[0]           \n"   \
  "vmov.u32    d11, #0              \n"

#define MID_COMPUTE_S1                    \
  "cmp %[cnt], #1                   \n"   \
  "vld1.8 {d12-d13}, [%[din_ptr0]]  \n"   \
  "vld1.8 {d16-d17}, [%[din_ptr1]]  \n"   \
  "vmov.u32 q12, #0                 \n"   \
  "vmov.u32 q13, #0                 \n"   \
  "vmov.u32 q14, #0                 \n"   \
  "vmov.u32 q15, #0                 \n"   \
  "blt 1f                           \n"   \
  "2:                               \n"   \
  /* r0 */                                \
  "vmull.s8 q10, d12, d2            \n"   \
  "vmull.s8 q11, d16, d2            \n"   \
  "vext.8   d14, d12, d13, #1       \n"   \
  "vext.8   d18, d16, d17, #1       \n"   \
  "vext.8   d15, d12, d13, #2       \n"   \
  "vext.8   d19, d16, d17, #2       \n"   \
  "vmlal.s8 q10, d14, d3            \n"   \
  "vmlal.s8 q11, d18, d3            \n"   \
  "vld1.8 {d12-d13}, [%[din_ptr2]]  \n"   \
  "add      %[din_ptr0], #8         \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8 q10, d15, d4            \n"   \
  "vmull.s8 q11, d19, d4            \n"   \
  /* r1 */                                \
  "vext.8   d14, d12, d13, #1       \n"   \
  "vext.8   d15, d12, d13, #2       \n"   \
  "vmlal.s8 q10, d16,  d5           \n"   \
  "vmlal.s8 q11, d12,  d5           \n"   \
  "add      %[din_ptr1], #8         \n"   \
  "vld1.8 {d16-d17}, [%[din_ptr3]]  \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8  q10, d18, d6           \n"   \
  "vmull.s8  q11, d14, d6           \n"   \
  "add      %[din_ptr2], #8         \n"   \
  "add      %[din_ptr3], #8         \n"   \
  "vmlal.s8  q10, d19, d7           \n"   \
  "vmlal.s8  q11, d15, d7           \n"   \
  /* r2 */                                \
  "vext.8   d18, d16, d17, #1       \n"   \
  "vext.8   d19, d16, d17, #2       \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8  q10, d12, d8           \n"   \
  "vmull.s8  q11, d16, d8           \n"   \
  "vmull.s8  q6,  d14, d9           \n"   \
  "vmull.s8  q8,  d18, d9           \n"   \
  "vmlal.s8  q10, d15, d10          \n"   \
  "vmlal.s8  q11, d19, d10         \n"    \
  "vaddw.s16 q12, q12, d12          \n"   \
  "vaddw.s16 q13, q13, d13          \n"   \
  "vaddw.s16 q14, q14, d16          \n"   \
  "vaddw.s16 q15, q15, d17          \n"   \
  "vld1.32   {d14-d15}, [%[vmax]]   \n"   \
  "vdup.32   q8, %[bias]            \n"   \
  "vdup.32   q9, %[bias]            \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vdup.32   q10, %[bias]           \n"   \
  "vdup.32   q11, %[bias]           \n"   \
  "vcvt.f32.s32   q12, q12          \n"   \
  "vcvt.f32.s32   q13, q13          \n"   \
  "vcvt.f32.s32   q14, q14          \n"   \
  "vcvt.f32.s32   q15, q15          \n"   \
  "vmov.u32 q6, #0                  \n"   \
  "subs %[cnt], #1                  \n"   \
  "vmla.f32 q8,   q12,  q7          \n"   \
  "vmla.f32 q9,   q13,  q7          \n"   \
  "vmla.f32 q10,  q14,  q7          \n"   \
  "vmla.f32 q11,  q15,  q7          \n"

#define MID_STORE_FLOAT                   \
  "vld1.8 {d12-d13}, [%[din_ptr0]]  \n"   \
  "vmov.u32 q12, #0                 \n"   \
  "vmov.u32 q13, #0                 \n"   \
  "vst1.32 {d16-d19}, [%[dout_ptr0]]!\n"  \
  "vld1.8 {d16-d17}, [%[din_ptr1]]  \n"   \
  "vmov.u32 q14, #0                 \n"   \
  "vmov.u32 q15, #0                 \n"   \
  "vst1.32 {d20-d23}, [%[dout_ptr1]]!\n"  \
  "bne  2b\n"

#define MID_STORE_INT8                    \
  "vld1.8 {d12-d13}, [%[din_ptr0]]  \n"   \
  "vld1.8 {d16-d17}, [%[din_ptr1]]  \n"   \
  "vmov.u32 q12, #0                 \n"   \
  "vmov.u32 q13, #0                 \n"   \
  "vmov.u32 q14, #0                 \n"   \
  "vmov.u32 q15, #0                 \n"   \
  "vst1.32 {d20}, [%[dout_ptr0]]!   \n"   \
  "vst1.32 {d21}, [%[dout_ptr1]]!   \n"   \
  "bne  2b\n"

#define RIGHT_COMPUTE_S1                  \
  "1:                               \n"   \
  "vld1.8 {d15}, [%[vmask]]     \n"   \
  "sub %[din_ptr0], %[right_pad_num_in]\n"\
  "sub %[din_ptr1], %[right_pad_num_in]\n"\
  "sub %[din_ptr2], %[right_pad_num_in]\n"\
  "sub %[din_ptr3], %[right_pad_num_in]\n"\
  "vld1.8 {d12-d13}, [%[din_ptr0]]  \n"   \
  "vld1.8 {d16-d17}, [%[din_ptr1]]  \n"   \
  "sub %[dout_ptr0], %[right_pad_num_out]\n"\
  "sub %[dout_ptr1], %[right_pad_num_out]\n"\
  /* r0 */                                \
  "vbif.s8  d13, d11, d15           \n"   \
  "vbif.s8  d17, d11, d15           \n"   \
  "vmull.s8 q10, d12, d2            \n"   \
  "vmull.s8 q11, d16, d2            \n"   \
  "vext.8   d14, d12, d13, #1       \n"   \
  "vext.8   d18, d16, d17, #1       \n"   \
  "vext.8   d15, d12, d13, #2       \n"   \
  "vext.8   d19, d16, d17, #2       \n"   \
  "vmlal.s8 q10, d14, d3            \n"   \
  "vmlal.s8 q11, d18, d3            \n"   \
  "vld1.8 {d12-d13}, [%[din_ptr2]]  \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vmull.s8 q10, d15, d4            \n"   \
  "vld1.8 {d15}, [%[vmask]]     \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8 q11, d19, d4            \n"   \
  "vbif.s8  d13, d11, d15           \n"   \
  /* r1 */                                \
  "vext.8   d14, d12, d13, #1       \n"   \
  "vext.8   d15, d12, d13, #2       \n"   \
  "vmlal.s8 q10, d16,  d5           \n"   \
  "vmlal.s8 q11, d12,  d5           \n"   \
  "vld1.8 {d16-d17}, [%[din_ptr3]]  \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vmull.s8  q10, d18, d6           \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8  q11, d14, d6           \n"   \
  "vmlal.s8  q10, d19, d7           \n"   \
  "vld1.8 {d19}, [%[vmask]]     \n"   \
  "vmlal.s8  q11, d15, d7           \n"   \
  "vbif.s8  d17, d11, d19           \n"   \
  /* r2 */                                \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vext.8   d18, d16, d17, #1       \n"   \
  "vext.8   d19, d16, d17, #2       \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vmull.s8  q10, d12, d8           \n"   \
  "vmull.s8  q11, d16, d8           \n"   \
  "vmull.s8  q6,  d14, d9           \n"   \
  "vmull.s8  q8,  d18, d9           \n"   \
  "vmlal.s8  q10, d15, d10          \n"   \
  "vmlal.s8  q11, d19, d10         \n"    \
  "vaddw.s16 q12, q12, d12          \n"   \
  "vaddw.s16 q13, q13, d13          \n"   \
  "vaddw.s16 q14, q14, d16          \n"   \
  "vaddw.s16 q15, q15, d17          \n"   \
  "vld1.32   {d14-d15}, [%[vmax]]   \n"   \
  "vdup.32   q8, %[bias]            \n"   \
  "vdup.32   q9, %[bias]            \n"   \
  "vaddw.s16 q12, q12, d20          \n"   \
  "vaddw.s16 q13, q13, d21          \n"   \
  "vaddw.s16 q14, q14, d22          \n"   \
  "vaddw.s16 q15, q15, d23          \n"   \
  "vdup.32   q10, %[bias]           \n"   \
  "vdup.32   q11, %[bias]           \n"   \
  "vcvt.f32.s32   q12, q12          \n"   \
  "vcvt.f32.s32   q13, q13          \n"   \
  "vcvt.f32.s32   q14, q14          \n"   \
  "vcvt.f32.s32   q15, q15          \n"   \
  "vmov.u32 q6, #0                  \n"   \
  "vmla.f32 q8,   q12,  q7          \n"   \
  "vmla.f32 q9,   q13,  q7          \n"   \
  "vmla.f32 q10,  q14,  q7          \n"   \
  "vmla.f32 q11,  q15,  q7          \n"
#endif
// clang-format on

template <typename Dtype>
void conv_depthwise_3x3s1_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               int flag_act,
                               float* alpha,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               ARMContext* ctx) {
  int threads = ctx->threads();
  int llc_size = ctx->llc_size() / 4;

  const int hout_c_block = 8;
  const int hout_r_kernel = 1;
  const int wout_block = 4;
  const int wout_round = ((wout + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round + 2;

  //! get h block
  //! llc_size = threads * win_round *  hout_c_block * hin_r_block *
  //! sizeof(int8_t)
  //!  + wout_round * hout_c_block * hout_r_block * threads * sizeof(int32_t)
  //! win_round = wout_round + 2
  //! hin_r_block = hout_r_block + 2
  int hout_r_block = (llc_size - 2 * win_round * threads * hout_c_block) /
                     (win_round * threads * hout_c_block +
                      hout_c_block * wout_round * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block =
      ((hout_r_block + hout_r_kernel - 1) / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block + 2;

  auto tmp_work_space = ctx->workspace_data<int8_t>();
  int8_t ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(int8_t) * win_round);
  Dtype ptr_write[wout_round];  // NOLINT

  int in_len = win_round * hout_c_block;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  int8_t* tmp_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 9;  // kernel_w * kernel_h;

  int ws = -padw;
  int we = ws + win_round;
  int w_loop = wout_round / 4;
  int chout = chin;

  int out_row_stride = hout_c_block * wout_round;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch = reinterpret_cast<int8_t*>(dout) +
                         n * chout * size_out_channel * sizeof(Dtype);
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h - padh;
      int he = hs + h_kernel + 2;

      LITE_PARALLEL_COMMON_BEGIN(c, tid, chout, 0, hout_c_block) {
#ifdef LITE_USE_THREAD_POOL
        int8_t* pre_din = tmp_din + tid * (pre_in_size + pre_out_size * 4);
        int32_t* pre_out = reinterpret_cast<int*>(pre_din + pre_in_size);
#elif defined(ARM_WITH_OMP)
        int8_t* pre_din =
            tmp_din + omp_get_thread_num() * (pre_in_size + pre_out_size * 4);
        int32_t* pre_out = reinterpret_cast<int*>(pre_din + pre_in_size);
#else
        int32_t* pre_out = reinterpret_cast<int32_t*>(tmp_din + pre_in_size);
        auto pre_din = tmp_din;
#endif
        prepack_input_nxwc8_int8_dw(
            din_batch, pre_din, c, hs, he, ws, we, chin, win, hin);

        const int8_t* block_inr0 = pre_din;
        const int8_t* block_inr1 = block_inr0 + in_len;
        const int8_t* block_inr2 = block_inr1 + in_len;

        const int8_t* weight_c = weights + c * w_stride;
#ifdef __aarch64__
        int8x8_t vw0 = vld1_s8(weight_c);
        int8x8_t vw1 = vld1_s8(weight_c + 8);
        int8x8_t vw2 = vld1_s8(weight_c + 16);
        int8x8_t vw3 = vld1_s8(weight_c + 24);
        int8x8_t vw4 = vld1_s8(weight_c + 32);
        int8x8_t vw5 = vld1_s8(weight_c + 40);
        int8x8_t vw6 = vld1_s8(weight_c + 48);
        int8x8_t vw7 = vld1_s8(weight_c + 56);
        int8x8_t vw8 = vld1_s8(weight_c + 64);
#endif
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          int cnt = w_loop;
          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;
          int32_t* ptr_out0 = pre_out + hk * out_row_stride;
#ifdef __aarch64__
          asm volatile(
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r0]], #32\n"
              "1:\n"
              /* inr0 -> outr0 */
              "ldp  d4, d5, [%[r0]]\n"           /* load r0, 4 */
              "smull v20.8h, v0.8b,  %[w0].8b\n" /* int16, out0 */
              "smull v21.8h, v1.8b,  %[w0].8b\n" /* int16, out1 */
              "smull v22.8h, v2.8b,  %[w0].8b\n" /* int16, out2 */
              "smull v23.8h, v3.8b,  %[w0].8b\n" /* int16, out3 */
              "smlal v20.8h, v1.8b,  %[w1].8b\n" /* int16, out0 */
              "smlal v21.8h, v2.8b,  %[w1].8b\n" /* int16, out1 */
              "smlal v22.8h, v3.8b,  %[w1].8b\n" /* int16, out2 */
              "smlal v23.8h, v4.8b,  %[w1].8b\n" /* int16, out3 */
              "ldp  d0, d1, [%[r1]], #16\n"      /* load r1, 0,1 */
              "sxtl  v24.4s, v20.4h\n"
              "sxtl2 v25.4s, v20.8h\n"
              "sxtl  v26.4s, v21.4h\n"
              "sxtl2 v27.4s, v21.8h\n"
              "sxtl  v28.4s, v22.4h\n"
              "sxtl2 v29.4s, v22.8h\n"
              "sxtl  v30.4s, v23.4h\n"
              "sxtl2 v31.4s, v23.8h\n"
              "smull v20.8h, v2.8b,  %[w2].8b\n" /* int16, out0 */
              "smull v21.8h, v3.8b,  %[w2].8b\n" /* int16, out1 */
              "smull v22.8h, v4.8b,  %[w2].8b\n" /* int16, out2 */
              "smull v23.8h, v5.8b,  %[w2].8b\n" /* int16, out3 */
              "ldp  d2, d3, [%[r1]], #16\n"      /* load r1, 2,3 */
              "smlal v20.8h, v0.8b,  %[w3].8b\n" /* int16, out0 */
              "smlal v21.8h, v1.8b,  %[w3].8b\n" /* int16, out1 */
              "smlal v22.8h, v2.8b,  %[w3].8b\n" /* int16, out2 */
              "smlal v23.8h, v3.8b,  %[w3].8b\n" /* int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldp  d4, d5, [%[r1]]\n" /* load r1, 4,5 */
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v1.8b,  %[w4].8b\n" /* int16, out0 */
              "smull v21.8h, v2.8b,  %[w4].8b\n" /* int16, out1 */
              "smull v22.8h, v3.8b,  %[w4].8b\n" /* int16, out1 */
              "smull v23.8h, v4.8b,  %[w4].8b\n" /* int16, out1 */
              "ldp  d0, d1, [%[r2]], #16\n"      /* load r2, 0,1 */
              "smlal v20.8h, v2.8b,  %[w5].8b\n" /* int16, out0 */
              "smlal v21.8h, v3.8b,  %[w5].8b\n" /* int16, out1 */
              "smlal v22.8h, v4.8b,  %[w5].8b\n" /* int16, out2 */
              "smlal v23.8h, v5.8b,  %[w5].8b\n" /* int16, out3 */
              "ldp  d2, d3, [%[r2]], #16\n"      /* load r2, 2,3 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldp  d4, d5, [%[r2]]\n" /* load r2 */
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v0.8b,  %[w6].8b\n" /* int16, out0 */
              "smull v21.8h, v1.8b,  %[w6].8b\n" /* int16, out1 */
              "smull v22.8h, v2.8b,  %[w6].8b\n" /* int16, out1 */
              "smull v23.8h, v3.8b,  %[w6].8b\n" /* int16, out1 */
              "smlal v20.8h, v1.8b,  %[w7].8b\n" /* int16, out0 */
              "smlal v21.8h, v2.8b,  %[w7].8b\n" /* int16, out1 */
              "smlal v22.8h, v3.8b,  %[w7].8b\n" /* int16, out1 */
              "smlal v23.8h, v4.8b,  %[w7].8b\n" /* int16, out1 */
              "ldp  d0, d1, [%[r0]], #16\n"      /* load r0, 0,1 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v2.8b,  %[w8].8b\n" /* int16, out0 */
              "smull v21.8h, v3.8b,  %[w8].8b\n" /* int16, out1 */
              "smull v22.8h, v4.8b,  %[w8].8b\n" /* int16, out1 */
              "smull v23.8h, v5.8b,  %[w8].8b\n" /* int16, out1 */
              "ldp  d2, d3, [%[r0]], #16\n"      /* load r0, 2,3 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "stp    q24, q25, [%[ptr_out0]], #32\n"
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "stp    q26, q27, [%[ptr_out0]], #32\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "subs    %w[cnt], %w[cnt], #1\n"
              "stp    q28, q29, [%[ptr_out0]], #32\n"
              "stp    q30, q31, [%[ptr_out0]], #32\n"
              "bne    1b\n"
              : [cnt] "+r"(cnt),
                [r0] "+r"(inr0),
                [r1] "+r"(inr1),
                [r2] "+r"(inr2),
                [ptr_out0] "+r"(ptr_out0)
              : [w0] "w"(vw0),
                [w1] "w"(vw1),
                [w2] "w"(vw2),
                [w3] "w"(vw3),
                [w4] "w"(vw4),
                [w5] "w"(vw5),
                [w6] "w"(vw6),
                [w7] "w"(vw7),
                [w8] "w"(vw8)
              : "cc",
                "memory",
                "v0",
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "v20",
                "v21",
                "v22",
                "v23",
                "v24",
                "v25",
                "v26",
                "v27",
                "v28",
                "v29",
                "v30",
                "v31"

              );
#else
          auto wptr = weight_c;
          asm volatile(
              "vld1.32    {d0-d3}, [%[r0]]!\n"   /* load r0, 0-4 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n" /* load w0-w1 */
              "1:\n"
              /* inr0 -> outr0 */
              "vld1.32    {d4-d5}, [%[r0]]\n"   /* load r0, 5-6 */
              "vmull.s8 q4, d0,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d1,  d6\n"          /* int16, out1 */
              "vmull.s8 q6, d2,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d3,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w2 */
              "vmlal.s8 q4, d1,  d7\n"          /* int16, out0 */
              "vmlal.s8 q5, d2,  d7\n"          /* int16, out1 */
              "vmlal.s8 q6, d3,  d7\n"          /* int16, out2 */
              "vmlal.s8 q7, d4,  d7\n"          /* int16, out3 */
              "vld1.32    {d7},   [%[wptr]]!\n" /* load w3 */
              "vmovl.s16  q8, d8\n"
              "vmovl.s16  q9, d9\n"
              "vmovl.s16  q10, d10\n"
              "vmovl.s16  q11, d11\n"
              "vld1.32    {d0-d1}, [%[r1]]!\n" /* load r1, 0-1 */
              "vmovl.s16  q12, d12\n"
              "vmovl.s16  q13, d13\n"
              "vmovl.s16  q14, d14\n"
              "vmovl.s16  q15, d15\n"
              "vmull.s8 q4, d2,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d3,  d6\n"          /* int16, out1 */
              "vld1.32    {d2-d3}, [%[r1]]!\n"  /* load r1, 2-3 */
              "vmull.s8 q6, d4,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d5,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w4 */
              /* inr1 -> outr0 */
              "vmlal.s8 q4, d0,  d7\n"        /* int16, out0 */
              "vmlal.s8 q5, d1,  d7\n"        /* int16, out1 */
              "vmlal.s8 q6, d2,  d7\n"        /* int16, out2 */
              "vmlal.s8 q7, d3,  d7\n"        /* int16, out3 */
              "vld1.32    {d4-d5}, [%[r1]]\n" /* load r1, 4-5 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d7},   [%[wptr]]!\n" /* load w5 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "vmull.s8 q4, d1,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d2,  d6\n"          /* int16, out1 */
              "vmull.s8 q6, d3,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d4,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w6 */
              "vld1.32    {d0-d1}, [%[r2]]!\n"  /* load r2, 0-1 */
              "vmlal.s8 q4, d2,  d7\n"          /* int16, out0 */
              "vmlal.s8 q5, d3,  d7\n"          /* int16, out1 */
              "vmlal.s8 q6, d4,  d7\n"          /* int16, out2 */
              "vmlal.s8 q7, d5,  d7\n"          /* int16, out3 */
              "vld1.32    {d7},   [%[wptr]]!\n" /* load w7 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d2-d3}, [%[r2]]!\n" /* load r2, 2-3 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "vld1.32    {d4-d5}, [%[r2]]\n" /* load r2, 4-5 */
              /* inr2 -> outr0 */
              "vmull.s8 q4, d0,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d1,  d6\n"          /* int16, out1 */
              "vmull.s8 q6, d2,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d3,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w8 */
              "vmlal.s8 q4, d1,  d7\n"          /* int16, out0 */
              "vmlal.s8 q5, d2,  d7\n"          /* int16, out1 */
              "vmlal.s8 q6, d3,  d7\n"          /* int16, out2 */
              "vmlal.s8 q7, d4,  d7\n"          /* int16, out3 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d0-d1}, [%[r0]]!\n" /* load r0, 0-1 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "sub %[wptr],   %[wptr],    #72\n"
              "vmull.s8 q4, d2,  d6\n"         /* int16, out0 */
              "vmull.s8 q5, d3,  d6\n"         /* int16, out1 */
              "vmull.s8 q6, d4,  d6\n"         /* int16, out2 */
              "vmull.s8 q7, d5,  d6\n"         /* int16, out3 */
              "vld1.32    {d2-d3}, [%[r0]]!\n" /* load r0, 2-3 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vst1.32    {d16-d19},  [%[ptr_out0]]!\n"
              "vld1.32    {d6-d7},   [%[wptr]]!\n" /* load w0-w1 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vst1.32    {d20-d23},  [%[ptr_out0]]!\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "subs    %[cnt], #1\n"
              "vst1.32    {d24-d27},  [%[ptr_out0]]!\n"
              "vst1.32    {d28-d31},  [%[ptr_out0]]!\n"
              "bne    1b\n"
              : [cnt] "+r"(cnt),
                [r0] "+r"(inr0),
                [r1] "+r"(inr1),
                [r2] "+r"(inr2),
                [ptr_out0] "+r"(ptr_out0),
                [wptr] "+r"(wptr)
              :
              : "cc",
                "memory",
                "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "q9",
                "q10",
                "q11",
                "q12",
                "q13",
                "q14",
                "q15");
#endif
          block_inr0 = block_inr1;
          block_inr1 = block_inr2;
          block_inr2 = block_inr1 + in_len;
        }
        write_int32_nchwc8_to_nchw<Dtype>(pre_out,
                                          reinterpret_cast<Dtype*>(dout_batch),
                                          c,
                                          c + hout_c_block,
                                          h,
                                          h + h_kernel,
                                          0,
                                          wout_round,
                                          chout,
                                          hout,
                                          wout,
                                          flag_act,
                                          alpha,
                                          bias + c,
                                          flag_bias,
                                          ptr_write,
                                          scale + c);
      }
      LITE_PARALLEL_END();
    }
  }
}

inline std::pair<uint32_t, uint32_t> right_mask_3x3s1p1_int8(int w_in,
                                                             int w_out,
                                                             uint8_t* vmask) {
  const uint8_t right_pad_idx[8] = {8, 9, 10, 11, 12, 13, 14, 15};
  uint32_t cnt_col = ((w_out >> 3) - 2);
  uint8_t size_right_remain = static_cast<uint8_t>(w_in - (7 + cnt_col * 8));
  if (size_right_remain >= 9) {
    cnt_col++;
    size_right_remain -= 8;
  }
  uint32_t cnt_remain = (size_right_remain == 8 && w_out % 8 == 0)
                            ? 8
                            : static_cast<uint32_t>(w_out % 8);
  size_right_remain = size_right_remain + 8 - cnt_remain;
  uint8x8_t vmask_rp2 =
      vcgt_u8(vdup_n_u8(size_right_remain), vld1_u8(right_pad_idx));
  vst1_u8(vmask, vmask_rp2);
  return std::make_pair(cnt_col, cnt_remain);
}

inline std::pair<uint32_t, uint32_t> right_mask_3x3s1p0_int8(int w_in,
                                                             int w_out,
                                                             uint8_t* vmask) {
  const uint8_t right_pad_idx[8] = {8, 9, 10, 11, 12, 13, 14, 15};
  uint32_t cnt_col = ((w_out >> 3) - 1);
  uint8_t size_right_remain = static_cast<uint8_t>(w_in - cnt_col * 8);
  if (size_right_remain >= 9) {
    cnt_col++;
    size_right_remain -= 8;
  }
  uint32_t cnt_remain = (size_right_remain == 8 && w_out % 8 == 0)
                            ? 8
                            : static_cast<uint32_t>(w_out % 8);
  size_right_remain = size_right_remain + 8 - cnt_remain;
  uint8x8_t vmask_rp2 =
      vcgt_u8(vdup_n_u8(size_right_remain), vld1_u8(right_pad_idx));
  vst1_u8(vmask, vmask_rp2);
  return std::make_pair(cnt_col, cnt_remain);
}

#define INIT_PTR_3x3_S1_INT8(Dtype, din, w_in) \
  Dtype* doutr0 = nullptr;                     \
  Dtype* doutr1 = nullptr;                     \
  const int8_t* dr0 = din;                     \
  const int8_t* dr1 = dr0 + w_in;              \
  const int8_t* dr2 = dr1 + w_in;              \
  const int8_t* dr3 = dr2 + w_in;              \
  const int8_t* din_ptr0 = nullptr;            \
  const int8_t* din_ptr1 = nullptr;            \
  const int8_t* din_ptr2 = nullptr;            \
  const int8_t* din_ptr3 = nullptr;

#define ASSIGN_PTR_3x3_S1_INT8(w_out) \
  din_ptr0 = dr0;                     \
  din_ptr1 = dr1;                     \
  din_ptr2 = dr2;                     \
  din_ptr3 = dr3;                     \
  doutr0 = dout_ptr;                  \
  doutr1 = doutr0 + w_out;

#define TOP_BOTTOM_BORDER_3x3_S1P1_INT8(w_in, h_in, h_out) \
  if (i == 0) {                                            \
    din_ptr0 = zero_ptr;                                   \
    din_ptr1 = dr0;                                        \
    din_ptr2 = dr1;                                        \
    din_ptr3 = dr2;                                        \
    dr0 = dr1;                                             \
    dr1 = dr2;                                             \
    dr2 = dr3;                                             \
    dr3 = dr2 + w_in;                                      \
  } else {                                                 \
    dr0 = dr2;                                             \
    dr1 = dr3;                                             \
    dr2 = dr1 + w_in;                                      \
    dr3 = dr2 + w_in;                                      \
  }                                                        \
  if (i + 3 > h_in) {                                      \
    switch (i + 3 - h_in) {                                \
      case 3:                                              \
        din_ptr1 = zero_ptr;                               \
      case 2:                                              \
        din_ptr2 = zero_ptr;                               \
      case 1:                                              \
        din_ptr3 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 2 > h_out) {                                     \
    doutr1 = write_ptr;                                    \
  }

#define TOP_BOTTOM_BORDER_3x3_S1P0_INT8(w_in, h_in, h_out) \
  dr0 = dr2;                                               \
  dr1 = dr3;                                               \
  dr2 = dr1 + w_in;                                        \
  dr3 = dr2 + w_in;                                        \
  if (i + 3 > h_in) {                                      \
    switch (i + 3 - h_in) {                                \
      case 3:                                              \
        din_ptr1 = zero_ptr;                               \
      case 2:                                              \
        din_ptr2 = zero_ptr;                               \
      case 1:                                              \
        din_ptr3 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 2 > h_out) {                                     \
    doutr1 = write_ptr;                                    \
  }

#define PARAM1                                                           \
  [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
      [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3),              \
      [dout_ptr0] "+r"(doutr0), [dout_ptr1] "+r"(doutr1)

#ifdef __aarch64__
#define FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, max_val, alpha) \
  int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);                                     \
  int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);                                     \
  int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);                                     \
  int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);                                     \
  int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);                                     \
  int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);                                     \
  int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);                                     \
  int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);                                     \
  int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);                                     \
  float32x4_t vscale = vdupq_n_f32(scale_val);                               \
  float vmax[8] = {                                                          \
      max_val, max_val, max_val, max_val, alpha, alpha, alpha, alpha};

#define PARAM2                                                        \
  [v0] "w"(wr00), [v1] "w"(wr01), [v2] "w"(wr02), [v3] "w"(wr10),     \
      [v4] "w"(wr11), [v5] "w"(wr12), [v6] "w"(wr20), [v7] "w"(wr21), \
      [v8] "w"(wr22), [bias] "r"(bias_val), [vscale] "w"(vscale),     \
      [vmask] "r"(vmask), [right_pad_num_out] "r"(right_pad_num_out), \
      [right_pad_num_in] "r"(right_pad_num_in)

#define ASM_PARAM                                                             \
  "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
      "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",   \
      "v21"

#else
#define FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, max_val, alpha) \
  float vmax[12] = {scale_val,                                               \
                    scale_val,                                               \
                    scale_val,                                               \
                    scale_val,                                               \
                    max_val,                                                 \
                    max_val,                                                 \
                    max_val,                                                 \
                    max_val,                                                 \
                    alpha,                                                   \
                    alpha,                                                   \
                    alpha,                                                   \
                    alpha};

#define PARAM2                                                            \
  [vmask] "r"(vmask), [bias] "r"(bias_val), [vmax] "r"(vmax),             \
      [wei_ptr] "r"(wei_ptr), [right_pad_num_out] "r"(right_pad_num_out), \
      [right_pad_num_in] "r"(right_pad_num_in)

#define ASM_PARAM                                                             \
  "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
      "q10", "q11", "q12", "q13", "q14", "q15"
#endif

template void conv_depthwise_3x3s1_int8<int8_t>(int8_t* dout,
                                                const int8_t* din,
                                                const int8_t* weights,
                                                const float* scale,
                                                const float* bias,
                                                bool flag_bias,
                                                int flag_act,
                                                float* alpha,
                                                int num,
                                                int chin,
                                                int hin,
                                                int win,
                                                int hout,
                                                int wout,
                                                int padw,
                                                int padh,
                                                ARMContext* ctx);

template void conv_depthwise_3x3s1_int8<float>(float* dout,
                                               const int8_t* din,
                                               const int8_t* weights,
                                               const float* scale,
                                               const float* bias,
                                               bool flag_bias,
                                               int flag_act,
                                               float* alpha,
                                               int num,
                                               int chin,
                                               int hin,
                                               int win,
                                               int hout,
                                               int wout,
                                               int padw,
                                               int padh,
                                               ARMContext* ctx);

void conv_depthwise_3x3s1p1_bias_int8_float(float* dout,
                                            const int8_t* din,
                                            const int8_t* weights,
                                            const float* scale,
                                            const float* bias,
                                            bool flag_bias,
                                            int flag_act,
                                            float* alpha,
                                            int num,
                                            int ch_in,
                                            int h_in,
                                            int w_in,
                                            int h_out,
                                            int w_out,
                                            ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p1_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in + 16);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(float, din_ch_ptr, w_in)

          for (int i = 0; i < h_in; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P1_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 LEFT_STORE_FLOAT MID_COMPUTE_S1
                         MID_STORE_FLOAT RIGHT_COMPUTE_S1 LEFT_STORE_FLOAT
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#else
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 LEFT_STORE_FLOAT MID_COMPUTE_S1
                         MID_STORE_FLOAT RIGHT_COMPUTE_S1 LEFT_STORE_FLOAT
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p1_bias_int8_int8(int8_t* dout,
                                           const int8_t* din,
                                           const int8_t* weights,
                                           const float* scale,
                                           const float* bias,
                                           bool flag_bias,
                                           int flag_act,
                                           float* alpha,
                                           int num,
                                           int ch_in,
                                           int h_in,
                                           int w_in,
                                           int h_out,
                                           int w_out,
                                           ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p1_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  int8_t* write_ptr = ctx->workspace_data<int8_t>() + w_in + 16;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(int8_t, din_ch_ptr, w_in)

          for (int i = 0; i < h_in; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P1_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            INIT_S1 LEFT_COMPUTE_S1 RESULT_INT8_MAX RESULT_INT8 LEFT_STORE_INT8
                MID_COMPUTE_S1 RESULT_INT8_MAX RESULT_INT8 MID_STORE_INT8
                    RIGHT_COMPUTE_S1 RESULT_INT8_MAX RESULT_INT8 LEFT_STORE_INT8
            : PARAM1
            : [vmax] "r"(vmax), PARAM2
            : ASM_PARAM);
#else
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 RESULT_INT8 LEFT_STORE_INT8
                         MID_COMPUTE_S1 RESULT_INT8 MID_STORE_INT8
                             RIGHT_COMPUTE_S1 RESULT_INT8 LEFT_STORE_INT8
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p0_bias_int8_float(float* dout,
                                            const int8_t* din,
                                            const int8_t* weights,
                                            const float* scale,
                                            const float* bias,
                                            bool flag_bias,
                                            int flag_act,
                                            float* alpha,
                                            int num,
                                            int ch_in,
                                            int h_in,
                                            int w_in,
                                            int h_out,
                                            int w_out,
                                            ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p0_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in + 16);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(float, din_ch_ptr, w_in)

          for (int i = 0; i < h_out; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P0_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            MID_COMPUTE_S1 MID_STORE_FLOAT RIGHT_COMPUTE_S1 LEFT_STORE_FLOAT
            : PARAM1
            : PARAM2
            : ASM_PARAM);
#else
        asm volatile(
            INIT_P0 MID_COMPUTE_S1 MID_STORE_FLOAT
            RIGHT_COMPUTE_S1 LEFT_STORE_FLOAT
            : PARAM1
            : PARAM2
            : ASM_PARAM);
#endif
        // clang-format off
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p0_bias_int8_int8(int8_t* dout,
                                           const int8_t* din,
                                           const int8_t* weights,
                                           const float* scale,
                                           const float* bias,
                                           bool flag_bias,
                                           int flag_act,
                                           float* alpha,
                                           int num,
                                           int ch_in,
                                           int h_in,
                                           int w_in,
                                           int h_out,
                                           int w_out,
                                           ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p0_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  int8_t* write_ptr = ctx->workspace_data<int8_t>() + w_in + 16;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(int8_t, din_ch_ptr, w_in)

      for (int i = 0; i < h_out; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P0_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            MID_COMPUTE_S1 RESULT_INT8_MAX RESULT_INT8 MID_STORE_INT8
            RIGHT_COMPUTE_S1 RESULT_INT8_MAX RESULT_INT8 LEFT_STORE_INT8
            : PARAM1
            : [vmax] "r"(vmax), PARAM2
            : ASM_PARAM);
#else
        asm volatile(INIT_P0 MID_COMPUTE_S1 RESULT_INT8 MID_STORE_INT8
                     RIGHT_COMPUTE_S1 RESULT_INT8 LEFT_STORE_INT8
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p1_bias_relu_int8_float(float* dout,
                                                 const int8_t* din,
                                                 const int8_t* weights,
                                                 const float* scale,
                                                 const float* bias,
                                                 bool flag_bias,
                                                 int flag_act,
                                                 float* alpha,
                                                 int num,
                                                 int ch_in,
                                                 int h_in,
                                                 int w_in,
                                                 int h_out,
                                                 int w_out,
                                                 ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p1_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in + 16);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(float, din_ch_ptr, w_in)

          for (int i = 0; i < h_in; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P1_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 RELU LEFT_STORE_FLOAT
                     MID_COMPUTE_S1 RELU MID_STORE_FLOAT
                     RIGHT_COMPUTE_S1 RELU LEFT_STORE_FLOAT
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#else
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 RELU LEFT_STORE_FLOAT 
                     MID_COMPUTE_S1 RELU MID_STORE_FLOAT
                     RIGHT_COMPUTE_S1 RELU LEFT_STORE_FLOAT
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p1_bias_relu6_int8_float(float* dout,
                                                  const int8_t* din,
                                                  const int8_t* weights,
                                                  const float* scale,
                                                  const float* bias,
                                                  bool flag_bias,
                                                  int flag_act,
                                                  float* alpha,
                                                  int num,
                                                  int ch_in,
                                                  int h_in,
                                                  int w_in,
                                                  int h_out,
                                                  int w_out,
                                                  ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p1_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in + 16);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, alpha[0])
      INIT_PTR_3x3_S1_INT8(float, din_ch_ptr, w_in)

          for (int i = 0; i < h_in; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P1_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 RELU6 LEFT_STORE_FLOAT
                     MID_COMPUTE_S1 RELU6 MID_STORE_FLOAT
                     RIGHT_COMPUTE_S1 RELU6 LEFT_STORE_FLOAT
                     : PARAM1
                     : [vmax] "r"(vmax), PARAM2
                     : ASM_PARAM);
#else
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 RELU6 LEFT_STORE_FLOAT 
                     MID_COMPUTE_S1 RELU6 MID_STORE_FLOAT
                     RIGHT_COMPUTE_S1 RELU6 LEFT_STORE_FLOAT
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p1_bias_relu_int8_int8(int8_t* dout,
                                                const int8_t* din,
                                                const int8_t* weights,
                                                const float* scale,
                                                const float* bias,
                                                bool flag_bias,
                                                int flag_act,
                                                float* alpha,
                                                int num,
                                                int ch_in,
                                                int h_in,
                                                int w_in,
                                                int h_out,
                                                int w_out,
                                                ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p1_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  int8_t* write_ptr = ctx->workspace_data<int8_t>() + w_in + 16;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(int8_t, din_ch_ptr, w_in)

          for (int i = 0; i < h_in; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P1_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            INIT_S1 LEFT_COMPUTE_S1 RELU RESULT_INT8 LEFT_STORE_INT8
            MID_COMPUTE_S1 RELU RESULT_INT8 MID_STORE_INT8
            RIGHT_COMPUTE_S1 RELU RESULT_INT8 LEFT_STORE_INT8
            : PARAM1
            : [vmax] "r"(vmax), PARAM2
            : ASM_PARAM);
#else
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 RELU RESULT_INT8 LEFT_STORE_INT8
                     MID_COMPUTE_S1 RELU RESULT_INT8 MID_STORE_INT8
                     RIGHT_COMPUTE_S1 RELU RESULT_INT8 LEFT_STORE_INT8
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p1_bias_relu6_int8_int8(int8_t* dout,
                                                 const int8_t* din,
                                                 const int8_t* weights,
                                                 const float* scale,
                                                 const float* bias,
                                                 bool flag_bias,
                                                 int flag_act,
                                                 float* alpha,
                                                 int num,
                                                 int ch_in,
                                                 int h_in,
                                                 int w_in,
                                                 int h_out,
                                                 int w_out,
                                                 ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p1_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  int8_t* write_ptr = ctx->workspace_data<int8_t>() + w_in + 16;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, alpha[0])
      INIT_PTR_3x3_S1_INT8(int8_t, din_ch_ptr, w_in)

          for (int i = 0; i < h_in; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P1_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            INIT_S1 LEFT_COMPUTE_S1 RELU6 RESULT_INT8 LEFT_STORE_INT8
            MID_COMPUTE_S1 RELU6 RESULT_INT8 MID_STORE_INT8
            RIGHT_COMPUTE_S1 RELU6 RESULT_INT8 LEFT_STORE_INT8
            : PARAM1
            : [vmax] "r"(vmax), PARAM2
            : ASM_PARAM);
#else
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 RELU6 RESULT_INT8 LEFT_STORE_INT8
                     MID_COMPUTE_S1 RELU6 RESULT_INT8 MID_STORE_INT8
                     RIGHT_COMPUTE_S1 RELU6 RESULT_INT8 LEFT_STORE_INT8
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p0_bias_relu_int8_float(float* dout,
                                                 const int8_t* din,
                                                 const int8_t* weights,
                                                 const float* scale,
                                                 const float* bias,
                                                 bool flag_bias,
                                                 int flag_act,
                                                 float* alpha,
                                                 int num,
                                                 int ch_in,
                                                 int h_in,
                                                 int w_in,
                                                 int h_out,
                                                 int w_out,
                                                 ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p0_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in + 16);
  int threads = ctx->threads();

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(float, din_ch_ptr, w_in)

          for (int i = 0; i < h_out; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P0_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            MID_COMPUTE_S1 RELU MID_STORE_FLOAT 
            RIGHT_COMPUTE_S1 RELU LEFT_STORE_FLOAT
            : PARAM1
            : PARAM2
            : ASM_PARAM);
#else
        asm volatile(
            INIT_P0 MID_COMPUTE_S1 RELU MID_STORE_FLOAT
            RIGHT_COMPUTE_S1 RELU LEFT_STORE_FLOAT
            : PARAM1
            : PARAM2
            : ASM_PARAM);
#endif
        // clang-format off
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p0_bias_relu6_int8_float(float* dout,
                                                  const int8_t* din,
                                                  const int8_t* weights,
                                                  const float* scale,
                                                  const float* bias,
                                                  bool flag_bias,
                                                  int flag_act,
                                                  float* alpha,
                                                  int num,
                                                  int ch_in,
                                                  int h_in,
                                                  int w_in,
                                                  int h_out,
                                                  int w_out,
                                                  ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p0_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in + 16);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, alpha[0])
      INIT_PTR_3x3_S1_INT8(float, din_ch_ptr, w_in)

      for (int i = 0; i < h_out; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P0_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            MID_COMPUTE_S1 RELU6 MID_STORE_FLOAT 
            RIGHT_COMPUTE_S1 RELU6 LEFT_STORE_FLOAT
            : PARAM1
            : [vmax] "r"(vmax), PARAM2
            : ASM_PARAM);
#else
        asm volatile(
            INIT_P0 MID_COMPUTE_S1 RELU6 MID_STORE_FLOAT
            RIGHT_COMPUTE_S1 RELU6 LEFT_STORE_FLOAT
            : PARAM1
            : PARAM2
            : ASM_PARAM);
#endif
        // clang-format off
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p0_bias_relu_int8_int8(int8_t* dout,
                                                const int8_t* din,
                                                const int8_t* weights,
                                                const float* scale,
                                                const float* bias,
                                                bool flag_bias,
                                                int flag_act,
                                                float* alpha,
                                                int num,
                                                int ch_in,
                                                int h_in,
                                                int w_in,
                                                int h_out,
                                                int w_out,
                                                ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p0_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  int8_t* write_ptr = ctx->workspace_data<int8_t>() + w_in + 16;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, 0.f)
      INIT_PTR_3x3_S1_INT8(int8_t, din_ch_ptr, w_in)

      for (int i = 0; i < h_out; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P0_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            MID_COMPUTE_S1 RELU RESULT_INT8 MID_STORE_INT8
            RIGHT_COMPUTE_S1 RELU RESULT_INT8 LEFT_STORE_INT8
            : PARAM1
            : [vmax] "r"(vmax), PARAM2
            : ASM_PARAM);
#else
        asm volatile(INIT_P0 MID_COMPUTE_S1 RELU RESULT_INT8 MID_STORE_INT8
                     RIGHT_COMPUTE_S1 RELU RESULT_INT8 LEFT_STORE_INT8
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1p0_bias_relu6_int8_int8(int8_t* dout,
                                                 const int8_t* din,
                                                 const int8_t* weights,
                                                 const float* scale,
                                                 const float* bias,
                                                 bool flag_bias,
                                                 int flag_act,
                                                 float* alpha,
                                                 int num,
                                                 int ch_in,
                                                 int h_in,
                                                 int w_in,
                                                 int h_out,
                                                 int w_out,
                                                 ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, (w_in + 16) * sizeof(int8_t));
  uint8_t vmask[8];
  auto&& res = right_mask_3x3s1p0_int8(w_in, w_out, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;

  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : (8 - cnt_remain);
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : (8 - cnt_remain);

  int8_t* write_ptr = ctx->workspace_data<int8_t>() + w_in + 16;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      FILL_WEIGHTS_BIAS_INT8(wei_ptr, bias_val, scale_val, -127.f, alpha[0])
      INIT_PTR_3x3_S1_INT8(int8_t, din_ch_ptr, w_in)

          for (int i = 0; i < h_out; i += 2) {
        // clang-format off
        ASSIGN_PTR_3x3_S1_INT8(w_out)
        TOP_BOTTOM_BORDER_3x3_S1P0_INT8(w_in, h_in, h_out)
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            MID_COMPUTE_S1 RELU6 RESULT_INT8 MID_STORE_INT8
            RIGHT_COMPUTE_S1 RELU6 RESULT_INT8 LEFT_STORE_INT8
            : PARAM1
            : [vmax] "r"(vmax), PARAM2
            : ASM_PARAM);
#else
        asm volatile(INIT_P0 MID_COMPUTE_S1 RELU6 RESULT_INT8 MID_STORE_INT8
                     RIGHT_COMPUTE_S1 RELU6 RESULT_INT8 LEFT_STORE_INT8
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s1_int8_float_impl(float* dout,
                                          const int8_t* din,
                                          const int8_t* weights,
                                          const float* scale,
                                          const float* bias,
                                          bool flag_bias,
                                          int flag_act,
                                          float* alpha,
                                          int num,
                                          int chin,
                                          int hin,
                                          int win,
                                          int hout,
                                          int wout,
                                          int padw,
                                          int padh,
                                          ARMContext* ctx) {
  if (padw == 1 && padh == 1) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p1_bias_int8_float(dout,
                                             din,
                                             weights,
                                             scale,
                                             bias,
                                             flag_bias,
                                             flag_act,
                                             alpha,
                                             num,
                                             chin,
                                             hin,
                                             win,
                                             hout,
                                             wout,
                                             ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p1_bias_relu_int8_float(dout,
                                                  din,
                                                  weights,
                                                  scale,
                                                  bias,
                                                  flag_bias,
                                                  flag_act,
                                                  alpha,
                                                  num,
                                                  chin,
                                                  hin,
                                                  win,
                                                  hout,
                                                  wout,
                                                  ctx);
    } else if (flag_act == 2) {
      conv_depthwise_3x3s1p1_bias_relu6_int8_float(dout,
                                                   din,
                                                   weights,
                                                   scale,
                                                   bias,
                                                   flag_bias,
                                                   flag_act,
                                                   alpha,
                                                   num,
                                                   chin,
                                                   hin,
                                                   win,
                                                   hout,
                                                   wout,
                                                   ctx);
    }
  } else if (padw == 0 && padh == 0) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p0_bias_int8_float(dout,
                                             din,
                                             weights,
                                             scale,
                                             bias,
                                             flag_bias,
                                             flag_act,
                                             alpha,
                                             num,
                                             chin,
                                             hin,
                                             win,
                                             hout,
                                             wout,
                                             ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p0_bias_relu_int8_float(dout,
                                                  din,
                                                  weights,
                                                  scale,
                                                  bias,
                                                  flag_bias,
                                                  flag_act,
                                                  alpha,
                                                  num,
                                                  chin,
                                                  hin,
                                                  win,
                                                  hout,
                                                  wout,
                                                  ctx);
    } else if (flag_act == 2) {
      conv_depthwise_3x3s1p0_bias_relu6_int8_float(dout,
                                                   din,
                                                   weights,
                                                   scale,
                                                   bias,
                                                   flag_bias,
                                                   flag_act,
                                                   alpha,
                                                   num,
                                                   chin,
                                                   hin,
                                                   win,
                                                   hout,
                                                   wout,
                                                   ctx);
    }
  }
}

void conv_depthwise_3x3s1_int8_int8_impl(int8_t* dout,
                                         const int8_t* din,
                                         const int8_t* weights,
                                         const float* scale,
                                         const float* bias,
                                         bool flag_bias,
                                         int flag_act,
                                         float* alpha,
                                         int num,
                                         int chin,
                                         int hin,
                                         int win,
                                         int hout,
                                         int wout,
                                         int padw,
                                         int padh,
                                         ARMContext* ctx) {
  if (padw == 1 && padh == 1) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p1_bias_int8_int8(dout,
                                            din,
                                            weights,
                                            scale,
                                            bias,
                                            flag_bias,
                                            flag_act,
                                            alpha,
                                            num,
                                            chin,
                                            hin,
                                            win,
                                            hout,
                                            wout,
                                            ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p1_bias_relu_int8_int8(dout,
                                                 din,
                                                 weights,
                                                 scale,
                                                 bias,
                                                 flag_bias,
                                                 flag_act,
                                                 alpha,
                                                 num,
                                                 chin,
                                                 hin,
                                                 win,
                                                 hout,
                                                 wout,
                                                 ctx);
    } else if (flag_act == 2) {
      conv_depthwise_3x3s1p1_bias_relu6_int8_int8(dout,
                                                  din,
                                                  weights,
                                                  scale,
                                                  bias,
                                                  flag_bias,
                                                  flag_act,
                                                  alpha,
                                                  num,
                                                  chin,
                                                  hin,
                                                  win,
                                                  hout,
                                                  wout,
                                                  ctx);
    }
  } else if (padw == 0 && padh == 0) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p0_bias_int8_int8(dout,
                                            din,
                                            weights,
                                            scale,
                                            bias,
                                            flag_bias,
                                            flag_act,
                                            alpha,
                                            num,
                                            chin,
                                            hin,
                                            win,
                                            hout,
                                            wout,
                                            ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p0_bias_relu_int8_int8(dout,
                                                 din,
                                                 weights,
                                                 scale,
                                                 bias,
                                                 flag_bias,
                                                 flag_act,
                                                 alpha,
                                                 num,
                                                 chin,
                                                 hin,
                                                 win,
                                                 hout,
                                                 wout,
                                                 ctx);
    } else if (flag_act == 2) {
      conv_depthwise_3x3s1p0_bias_relu6_int8_int8(dout,
                                                  din,
                                                  weights,
                                                  scale,
                                                  bias,
                                                  flag_bias,
                                                  flag_act,
                                                  alpha,
                                                  num,
                                                  chin,
                                                  hin,
                                                  win,
                                                  hout,
                                                  wout,
                                                  ctx);
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
