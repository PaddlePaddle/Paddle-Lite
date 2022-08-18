// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
#include "lite/backends/arm/math/sve/conv3x3s2_direct_int8_sve.h"
#include <arm_sve.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve2 {
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
// sshllb/sshllt needs sve2 support
#define INIT_WEIGHT            \
  "ldr q10, [%x[wc0]]\n"       \
  "ldr q11, [%x[wc0], #8]\n"   \
  "ldr q12, [%x[wc0], #16]\n"  \
  "ldr q13, [%x[wc0], #24]\n"  \
  "ldr q14, [%x[wc0], #32]\n"  \
  "ldr q15, [%x[wc0], #40]\n"  \
  "ldr q16, [%x[wc0], #48]\n"  \
  "ldr q17, [%x[wc0], #56]\n"  \
  "ldr q18, [%x[wc0], #64]\n"  \
  "sshll v10.8h, v10.8b, #0\n" \
  "sshll v11.8h, v11.8b, #0\n" \
  "sshll v12.8h, v12.8b, #0\n" \
  "sshll v13.8h, v13.8b, #0\n" \
  "sshll v14.8h, v14.8b, #0\n" \
  "sshll v15.8h, v15.8b, #0\n" \
  "sshll v16.8h, v16.8b, #0\n" \
  "sshll v17.8h, v17.8b, #0\n" \
  "sshll v18.8h, v18.8b, #0\n"

#define COMPUTE_SVE2_1             \
  "ld1b z8.b, p0/Z, [%x[r0]]\n"    \
  "add %x[r0], %x[r0], x10\n"      \
  "ld1b z9.b, p0/Z, [%x[r2]]\n"    \
  "add %x[r2], %x[r2], x10\n"      \
  "sshllb z0.h, z8.b, #0\n"        \
  "sshllt z1.h, z8.b, #0\n"        \
  "sshllb z2.h, z9.b, #0\n"        \
  "sshllt z3.h, z9.b, #0\n"        \
  "smullb z19.s, z10.h, z0.h[0]\n" \
  "smullt z20.s, z10.h, z0.h[0]\n" \
  "smullb z21.s, z10.h, z0.h[1]\n" \
  "smullt z22.s, z10.h, z0.h[1]\n" \
  "smullb z23.s, z10.h, z0.h[2]\n" \
  "smullt z24.s, z10.h, z0.h[2]\n" \
  "smullb z25.s, z10.h, z0.h[3]\n" \
  "smullt z26.s, z10.h, z0.h[3]\n" \
  "smullb z27.s, z10.h, z0.h[4]\n" \
  "smullt z28.s, z10.h, z0.h[4]\n" \
  "smullb z29.s, z10.h, z0.h[5]\n" \
  "smullt z30.s, z10.h, z0.h[5]\n" \
  "smullb z6.s,  z10.h, z0.h[6]\n" \
  "smullt z7.s,  z10.h, z0.h[6]\n" \
  "smullb z8.s,  z10.h, z0.h[7]\n" \
  "smullt z9.s,  z10.h, z0.h[7]\n" \
  "smlalb z19.s, z16.h, z2.h[0]\n" \
  "smlalt z20.s, z16.h, z2.h[0]\n" \
  "smlalb z21.s, z16.h, z2.h[1]\n" \
  "smlalt z22.s, z16.h, z2.h[1]\n" \
  "smlalb z23.s, z16.h, z2.h[2]\n" \
  "smlalt z24.s, z16.h, z2.h[2]\n" \
  "smlalb z25.s, z16.h, z2.h[3]\n" \
  "smlalt z26.s, z16.h, z2.h[3]\n" \
  "smlalb z27.s, z16.h, z2.h[4]\n" \
  "smlalt z28.s, z16.h, z2.h[4]\n" \
  "smlalb z29.s, z16.h, z2.h[5]\n" \
  "smlalt z30.s, z16.h, z2.h[5]\n" \
  "smlalb z6.s,  z16.h, z2.h[6]\n" \
  "smlalt z7.s,  z16.h, z2.h[6]\n" \
  "smlalb z8.s,  z16.h, z2.h[7]\n" \
  "smlalt z9.s,  z16.h, z2.h[7]\n" \
  "smlalb z19.s, z11.h, z1.h[0]\n" \
  "smlalt z20.s, z11.h, z1.h[0]\n" \
  "smlalb z21.s, z11.h, z1.h[1]\n" \
  "smlalt z22.s, z11.h, z1.h[1]\n" \
  "smlalb z23.s, z11.h, z1.h[2]\n" \
  "smlalt z24.s, z11.h, z1.h[2]\n" \
  "smlalb z25.s, z11.h, z1.h[3]\n" \
  "smlalt z26.s, z11.h, z1.h[3]\n" \
  "smlalb z27.s, z11.h, z1.h[4]\n" \
  "smlalt z28.s, z11.h, z1.h[4]\n" \
  "smlalb z29.s, z11.h, z1.h[5]\n" \
  "smlalt z30.s, z11.h, z1.h[5]\n" \
  "smlalb z6.s,  z11.h, z1.h[6]\n" \
  "smlalt z7.s,  z11.h, z1.h[6]\n" \
  "smlalb z8.s,  z11.h, z1.h[7]\n" \
  "smlalt z9.s,  z11.h, z1.h[7]\n" \
  "smlalb z19.s, z17.h, z3.h[0]\n" \
  "smlalt z20.s, z17.h, z3.h[0]\n" \
  "smlalb z21.s, z17.h, z3.h[1]\n" \
  "smlalt z22.s, z17.h, z3.h[1]\n" \
  "smlalb z23.s, z17.h, z3.h[2]\n" \
  "smlalt z24.s, z17.h, z3.h[2]\n" \
  "smlalb z25.s, z17.h, z3.h[3]\n" \
  "smlalt z26.s, z17.h, z3.h[3]\n" \
  "smlalb z27.s, z17.h, z3.h[4]\n" \
  "smlalt z28.s, z17.h, z3.h[4]\n" \
  "smlalb z29.s, z17.h, z3.h[5]\n" \
  "smlalt z30.s, z17.h, z3.h[5]\n" \
  "smlalb z6.s,  z17.h, z3.h[6]\n" \
  "smlalt z7.s,  z17.h, z3.h[6]\n" \
  "smlalb z8.s,  z17.h, z3.h[7]\n" \
  "smlalt z9.s,  z17.h, z3.h[7]\n" \
  "ldr q31, [%x[r0]]\n"            \
  "sshllb z4.h, z31.b, #0\n"       \
  "smlalb z19.s, z12.h, z0.h[1]\n" \
  "smlalt z20.s, z12.h, z0.h[1]\n" \
  "smlalb z21.s, z12.h, z0.h[2]\n" \
  "smlalt z22.s, z12.h, z0.h[2]\n" \
  "smlalb z23.s, z12.h, z0.h[3]\n" \
  "smlalt z24.s, z12.h, z0.h[3]\n" \
  "smlalb z25.s, z12.h, z0.h[4]\n" \
  "smlalt z26.s, z12.h, z0.h[4]\n" \
  "smlalb z27.s, z12.h, z0.h[5]\n" \
  "smlalt z28.s, z12.h, z0.h[5]\n" \
  "smlalb z29.s, z12.h, z0.h[6]\n" \
  "smlalt z30.s, z12.h, z0.h[6]\n" \
  "smlalb z6.s,  z12.h, z0.h[7]\n" \
  "smlalt z7.s,  z12.h, z0.h[7]\n" \
  "smlalb z8.s,  z12.h, z4.h[0]\n" \
  "smlalt z9.s,  z12.h, z4.h[0]\n" \
  "ld1b z31.b, p0/Z, [%x[r1]]\n"   \
  "add %x[r1], %x[r1], x10\n"      \
  "sshllb z0.h, z31.b, #0\n"       \
  "sshllt z1.h, z31.b, #0\n"       \
  "smlalb z19.s, z18.h, z2.h[1]\n" \
  "smlalt z20.s, z18.h, z2.h[1]\n" \
  "smlalb z21.s, z18.h, z2.h[2]\n" \
  "smlalt z22.s, z18.h, z2.h[2]\n" \
  "smlalb z23.s, z18.h, z2.h[3]\n" \
  "smlalt z24.s, z18.h, z2.h[3]\n" \
  "smlalb z25.s, z18.h, z2.h[4]\n" \
  "smlalt z26.s, z18.h, z2.h[4]\n" \
  "smlalb z27.s, z18.h, z2.h[5]\n" \
  "smlalt z28.s, z18.h, z2.h[5]\n" \
  "smlalb z29.s, z18.h, z2.h[6]\n" \
  "smlalt z30.s, z18.h, z2.h[6]\n" \
  "smlalb z6.s,  z18.h, z2.h[7]\n" \
  "smlalt z7.s,  z18.h, z2.h[7]\n" \
  "ldr q31, [%x[r2]]\n"            \
  "sshllb z5.h, z31.b, #0\n"       \
  "smlalb z8.s, z18.h, z5.h[0]\n"  \
  "smlalt z9.s, z18.h, z5.h[0]\n"  \
  "smlalb z19.s, z13.h, z0.h[0]\n" \
  "smlalt z20.s, z13.h, z0.h[0]\n" \
  "smlalb z21.s, z13.h, z0.h[1]\n" \
  "smlalt z22.s, z13.h, z0.h[1]\n" \
  "smlalb z23.s, z13.h, z0.h[2]\n" \
  "smlalt z24.s, z13.h, z0.h[2]\n" \
  "smlalb z25.s, z13.h, z0.h[3]\n" \
  "smlalt z26.s, z13.h, z0.h[3]\n" \
  "smlalb z27.s, z13.h, z0.h[4]\n" \
  "smlalt z28.s, z13.h, z0.h[4]\n" \
  "smlalb z29.s, z13.h, z0.h[5]\n" \
  "smlalt z30.s, z13.h, z0.h[5]\n" \
  "smlalb z6.s,  z13.h, z0.h[6]\n" \
  "smlalt z7.s,  z13.h, z0.h[6]\n" \
  "smlalb z8.s,  z13.h, z0.h[7]\n" \
  "smlalt z9.s,  z13.h, z0.h[7]\n" \
  "smlalb z19.s, z14.h, z1.h[0]\n" \
  "smlalt z20.s, z14.h, z1.h[0]\n" \
  "smlalb z21.s, z14.h, z1.h[1]\n" \
  "smlalt z22.s, z14.h, z1.h[1]\n" \
  "smlalb z23.s, z14.h, z1.h[2]\n" \
  "smlalt z24.s, z14.h, z1.h[2]\n" \
  "smlalb z25.s, z14.h, z1.h[3]\n" \
  "smlalt z26.s, z14.h, z1.h[3]\n" \
  "smlalb z27.s, z14.h, z1.h[4]\n" \
  "smlalt z28.s, z14.h, z1.h[4]\n" \
  "smlalb z29.s, z14.h, z1.h[5]\n" \
  "smlalt z30.s, z14.h, z1.h[5]\n" \
  "smlalb z6.s,  z14.h, z1.h[6]\n" \
  "smlalt z7.s,  z14.h, z1.h[6]\n" \
  "smlalb z8.s,  z14.h, z1.h[7]\n" \
  "smlalt z9.s,  z14.h, z1.h[7]\n" \
  "ldr q31, [%x[r1]]\n"            \
  "sshllb z4.h, z31.b, #0\n"       \
  "smlalb z19.s, z15.h, z0.h[1]\n" \
  "smlalt z20.s, z15.h, z0.h[1]\n" \
  "smlalb z21.s, z15.h, z0.h[2]\n" \
  "smlalt z22.s, z15.h, z0.h[2]\n" \
  "smlalb z23.s, z15.h, z0.h[3]\n" \
  "smlalt z24.s, z15.h, z0.h[3]\n" \
  "smlalb z25.s, z15.h, z0.h[4]\n" \
  "smlalt z26.s, z15.h, z0.h[4]\n" \
  "smlalb z27.s, z15.h, z0.h[5]\n" \
  "smlalt z28.s, z15.h, z0.h[5]\n" \
  "smlalb z29.s, z15.h, z0.h[6]\n" \
  "smlalt z30.s, z15.h, z0.h[6]\n" \
  "smlalb z6.s,  z15.h, z0.h[7]\n" \
  "smlalt z7.s,  z15.h, z0.h[7]\n" \
  "smlalb z8.s,  z15.h, z4.h[0]\n" \
  "smlalt z9.s,  z15.h, z4.h[0]\n"

#define COMPUTE_SVE2_1_B4           \
  "ld1b z8.b, p0/Z, [%x[r0]]\n"     \
  "add %x[r0], %x[r0], x10\n"       \
  "ld1b z9.b, p0/Z, [%x[r2]]\n"     \
  "add %x[r2], %x[r2], x10\n"       \
  "sshllb z0.h, z8.b, #0\n"         \
  "sshllt z1.h, z8.b, #0\n"         \
  "sshllb z2.h, z9.b, #0\n"         \
  "sshllt z3.h, z9.b, #0\n"         \
  "smullb z19.s, z10.h, z0.h[0]\n"  \
  "smullt z20.s, z10.h, z0.h[0]\n"  \
  "smullb z21.s, z10.h, z0.h[1]\n"  \
  "smullt z22.s, z10.h, z0.h[1]\n"  \
  "smullb z23.s, z10.h, z0.h[2]\n"  \
  "smullt z24.s, z10.h, z0.h[2]\n"  \
  "smullb z25.s, z10.h, z0.h[3]\n"  \
  "smullt z26.s, z10.h, z0.h[3]\n"  \
  "smlalb z19.s, z16.h, z2.h[0]\n"  \
  "smlalt z20.s, z16.h, z2.h[0]\n"  \
  "smlalb z21.s, z16.h, z2.h[1]\n"  \
  "smlalt z22.s, z16.h, z2.h[1]\n"  \
  "smlalb z23.s, z16.h, z2.h[2]\n"  \
  "smlalt z24.s, z16.h, z2.h[2]\n"  \
  "smlalb z25.s, z16.h, z2.h[3]\n"  \
  "smlalt z26.s, z16.h, z2.h[3]\n"  \
  "smlalb z19.s, z11.h, z1.h[0]\n"  \
  "smlalt z20.s, z11.h, z1.h[0]\n"  \
  "smlalb z21.s, z11.h, z1.h[1]\n"  \
  "smlalt z22.s, z11.h, z1.h[1]\n"  \
  "smlalb z23.s, z11.h, z1.h[2]\n"  \
  "smlalt z24.s, z11.h, z1.h[2]\n"  \
  "smlalb z25.s, z11.h, z1.h[3]\n"  \
  "smlalt z26.s, z11.h, z1.h[3]\n"  \
  "smlalb z19.s, z17.h, z3.h[0]\n"  \
  "smlalt z20.s, z17.h, z3.h[0]\n"  \
  "smlalb z21.s, z17.h, z3.h[1]\n"  \
  "smlalt z22.s, z17.h, z3.h[1]\n"  \
  "smlalb z23.s, z17.h, z3.h[2]\n"  \
  "smlalt z24.s, z17.h, z3.h[2]\n"  \
  "smlalb z25.s, z17.h, z3.h[3]\n"  \
  "smlalt z26.s, z17.h, z3.h[3]\n"  \
  "ldr q31, [%x[r0]]\n"             \
  "sshllb z4.h, z31.b, #0\n"        \
  "smlalb z19.s, z12.h, z0.h[1]\n"  \
  "smlalt z20.s, z12.h, z0.h[1]\n"  \
  "smlalb z21.s, z12.h, z0.h[2]\n"  \
  "smlalt z22.s, z12.h, z0.h[2]\n"  \
  "smlalb z23.s, z12.h, z0.h[3]\n"  \
  "smlalt z24.s, z12.h, z0.h[3]\n"  \
  "smlalb z25.s,  z12.h, z4.h[0]\n" \
  "smlalt z26.s,  z12.h, z4.h[0]\n" \
  "ld1b z31.b, p0/Z, [%x[r1]]\n"    \
  "add %x[r1], %x[r1], x10\n"       \
  "sshllb z0.h, z31.b, #0\n"        \
  "sshllt z1.h, z31.b, #0\n"        \
  "smlalb z19.s, z18.h, z2.h[1]\n"  \
  "smlalt z20.s, z18.h, z2.h[1]\n"  \
  "smlalb z21.s, z18.h, z2.h[2]\n"  \
  "smlalt z22.s, z18.h, z2.h[2]\n"  \
  "smlalb z23.s, z18.h, z2.h[3]\n"  \
  "smlalt z24.s, z18.h, z2.h[3]\n"  \
  "ldr q31, [%x[r2]]\n"             \
  "sshllb z5.h, z31.b, #0\n"        \
  "smlalb z25.s, z18.h, z5.h[0]\n"  \
  "smlalt z26.s, z18.h, z5.h[0]\n"  \
  "smlalb z19.s, z13.h, z0.h[0]\n"  \
  "smlalt z20.s, z13.h, z0.h[0]\n"  \
  "smlalb z21.s, z13.h, z0.h[1]\n"  \
  "smlalt z22.s, z13.h, z0.h[1]\n"  \
  "smlalb z23.s, z13.h, z0.h[2]\n"  \
  "smlalt z24.s, z13.h, z0.h[2]\n"  \
  "smlalb z25.s, z13.h, z0.h[3]\n"  \
  "smlalt z26.s, z13.h, z0.h[3]\n"  \
  "smlalb z19.s, z14.h, z1.h[0]\n"  \
  "smlalt z20.s, z14.h, z1.h[0]\n"  \
  "smlalb z21.s, z14.h, z1.h[1]\n"  \
  "smlalt z22.s, z14.h, z1.h[1]\n"  \
  "smlalb z23.s, z14.h, z1.h[2]\n"  \
  "smlalt z24.s, z14.h, z1.h[2]\n"  \
  "smlalb z25.s, z14.h, z1.h[3]\n"  \
  "smlalt z26.s, z14.h, z1.h[3]\n"  \
  "ldr q31, [%x[r1]]\n"             \
  "sshllb z4.h, z31.b, #0\n"        \
  "smlalb z19.s, z15.h, z0.h[1]\n"  \
  "smlalt z20.s, z15.h, z0.h[1]\n"  \
  "smlalb z21.s, z15.h, z0.h[2]\n"  \
  "smlalt z22.s, z15.h, z0.h[2]\n"  \
  "smlalb z23.s, z15.h, z0.h[3]\n"  \
  "smlalt z24.s, z15.h, z0.h[3]\n"  \
  "smlalb z25.s,  z15.h, z4.h[0]\n" \
  "smlalt z26.s,  z15.h, z4.h[0]\n"

#define STORE_LINE0_BLK8                          \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z19.s, p0/M, z19.s, z0.s\n"                \
  "add z20.s, p0/M, z20.s, z1.s\n"                \
  "st2w {z19.s, z20.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z21.s, p0/M, z21.s, z0.s\n"                \
  "add z22.s, p0/M, z22.s, z1.s\n"                \
  "st2w {z21.s, z22.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z23.s, p0/M, z23.s, z0.s\n"                \
  "add z24.s, p0/M, z24.s, z1.s\n"                \
  "st2w {z23.s, z24.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z25.s, p0/M, z25.s, z0.s\n"                \
  "add z26.s, p0/M, z26.s, z1.s\n"                \
  "st2w {z25.s, z26.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z27.s, p0/M, z27.s, z0.s\n"                \
  "add z28.s, p0/M, z28.s, z1.s\n"                \
  "st2w {z27.s, z28.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z29.s, p0/M, z29.s, z0.s\n"                \
  "add z30.s, p0/M, z30.s, z1.s\n"                \
  "st2w {z29.s, z30.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z6.s, p0/M, z6.s, z0.s\n"                  \
  "add z7.s, p0/M, z7.s, z1.s\n"                  \
  "st2w {z6.s, z7.s}, p0, [%x[ptr_out0]]\n"       \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z8.s, p0/M, z8.s, z0.s\n"                  \
  "add z9.s, p0/M, z9.s, z1.s\n"                  \
  "st2w {z8.s, z9.s}, p0, [%x[ptr_out0]]\n"       \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n"

#define STORE_LINE0_BLK4                          \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z19.s, p0/M, z19.s, z0.s\n"                \
  "add z20.s, p0/M, z20.s, z1.s\n"                \
  "st2w {z19.s, z20.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z21.s, p0/M, z21.s, z0.s\n"                \
  "add z22.s, p0/M, z22.s, z1.s\n"                \
  "st2w {z21.s, z22.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z23.s, p0/M, z23.s, z0.s\n"                \
  "add z24.s, p0/M, z24.s, z1.s\n"                \
  "st2w {z23.s, z24.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out0]]\n"     \
  "add z25.s, p0/M, z25.s, z0.s\n"                \
  "add z26.s, p0/M, z26.s, z1.s\n"                \
  "st2w {z25.s, z26.s}, p0, [%x[ptr_out0]]\n"     \
  "add %x[ptr_out0], %x[ptr_out0], x10, lsl #1\n"

#define COMPUTE_SVE2_2             \
  "ld1b z31.b, p0/Z, [%x[r3]]\n"   \
  "add %x[r3], %x[r3], x10\n"      \
  "sshllb z0.h, z31.b, #0\n"       \
  "sshllt z1.h, z31.b, #0\n"       \
  "ldr q31, [%x[r3]]\n"            \
  "sshllb z4.h, z31.b, #0\n"       \
  "smullb z19.s, z10.h, z2.h[0]\n" \
  "smullt z20.s, z10.h, z2.h[0]\n" \
  "smullb z21.s, z10.h, z2.h[1]\n" \
  "smullt z22.s, z10.h, z2.h[1]\n" \
  "smullb z23.s, z10.h, z2.h[2]\n" \
  "smullt z24.s, z10.h, z2.h[2]\n" \
  "smullb z25.s, z10.h, z2.h[3]\n" \
  "smullt z26.s, z10.h, z2.h[3]\n" \
  "smullb z27.s, z10.h, z2.h[4]\n" \
  "smullt z28.s, z10.h, z2.h[4]\n" \
  "smullb z29.s, z10.h, z2.h[5]\n" \
  "smullt z30.s, z10.h, z2.h[5]\n" \
  "smullb z6.s,  z10.h, z2.h[6]\n" \
  "smullt z7.s,  z10.h, z2.h[6]\n" \
  "smullb z8.s,  z10.h, z2.h[7]\n" \
  "smullt z9.s,  z10.h, z2.h[7]\n" \
  "smlalb z19.s, z11.h, z3.h[0]\n" \
  "smlalt z20.s, z11.h, z3.h[0]\n" \
  "smlalb z21.s, z11.h, z3.h[1]\n" \
  "smlalt z22.s, z11.h, z3.h[1]\n" \
  "smlalb z23.s, z11.h, z3.h[2]\n" \
  "smlalt z24.s, z11.h, z3.h[2]\n" \
  "smlalb z25.s, z11.h, z3.h[3]\n" \
  "smlalt z26.s, z11.h, z3.h[3]\n" \
  "smlalb z27.s, z11.h, z3.h[4]\n" \
  "smlalt z28.s, z11.h, z3.h[4]\n" \
  "smlalb z29.s, z11.h, z3.h[5]\n" \
  "smlalt z30.s, z11.h, z3.h[5]\n" \
  "smlalb z6.s,  z11.h, z3.h[6]\n" \
  "smlalt z7.s,  z11.h, z3.h[6]\n" \
  "smlalb z8.s,  z11.h, z3.h[7]\n" \
  "smlalt z9.s,  z11.h, z3.h[7]\n" \
  "smlalb z19.s, z12.h, z2.h[1]\n" \
  "smlalt z20.s, z12.h, z2.h[1]\n" \
  "smlalb z21.s, z12.h, z2.h[2]\n" \
  "smlalt z22.s, z12.h, z2.h[2]\n" \
  "smlalb z23.s, z12.h, z2.h[3]\n" \
  "smlalt z24.s, z12.h, z2.h[3]\n" \
  "smlalb z25.s, z12.h, z2.h[4]\n" \
  "smlalt z26.s, z12.h, z2.h[4]\n" \
  "smlalb z27.s, z12.h, z2.h[5]\n" \
  "smlalt z28.s, z12.h, z2.h[5]\n" \
  "smlalb z29.s, z12.h, z2.h[6]\n" \
  "smlalt z30.s, z12.h, z2.h[6]\n" \
  "smlalb z6.s,  z12.h, z2.h[7]\n" \
  "smlalt z7.s,  z12.h, z2.h[7]\n" \
  "smlalb z8.s,  z12.h, z5.h[0]\n" \
  "smlalt z9.s,  z12.h, z5.h[0]\n" \
  "smlalb z19.s, z13.h, z0.h[0]\n" \
  "smlalt z20.s, z13.h, z0.h[0]\n" \
  "smlalb z21.s, z13.h, z0.h[1]\n" \
  "smlalt z22.s, z13.h, z0.h[1]\n" \
  "smlalb z23.s, z13.h, z0.h[2]\n" \
  "smlalt z24.s, z13.h, z0.h[2]\n" \
  "smlalb z25.s, z13.h, z0.h[3]\n" \
  "smlalt z26.s, z13.h, z0.h[3]\n" \
  "smlalb z27.s, z13.h, z0.h[4]\n" \
  "smlalt z28.s, z13.h, z0.h[4]\n" \
  "smlalb z29.s, z13.h, z0.h[5]\n" \
  "smlalt z30.s, z13.h, z0.h[5]\n" \
  "smlalb z6.s,  z13.h, z0.h[6]\n" \
  "smlalt z7.s,  z13.h, z0.h[6]\n" \
  "smlalb z8.s,  z13.h, z0.h[7]\n" \
  "smlalt z9.s,  z13.h, z0.h[7]\n" \
  "smlalb z19.s, z14.h, z1.h[0]\n" \
  "smlalt z20.s, z14.h, z1.h[0]\n" \
  "smlalb z21.s, z14.h, z1.h[1]\n" \
  "smlalt z22.s, z14.h, z1.h[1]\n" \
  "smlalb z23.s, z14.h, z1.h[2]\n" \
  "smlalt z24.s, z14.h, z1.h[2]\n" \
  "smlalb z25.s, z14.h, z1.h[3]\n" \
  "smlalt z26.s, z14.h, z1.h[3]\n" \
  "smlalb z27.s, z14.h, z1.h[4]\n" \
  "smlalt z28.s, z14.h, z1.h[4]\n" \
  "smlalb z29.s, z14.h, z1.h[5]\n" \
  "smlalt z30.s, z14.h, z1.h[5]\n" \
  "smlalb z6.s,  z14.h, z1.h[6]\n" \
  "smlalt z7.s,  z14.h, z1.h[6]\n" \
  "smlalb z8.s,  z14.h, z1.h[7]\n" \
  "smlalt z9.s,  z14.h, z1.h[7]\n" \
  "smlalb z19.s, z15.h, z0.h[1]\n" \
  "smlalt z20.s, z15.h, z0.h[1]\n" \
  "smlalb z21.s, z15.h, z0.h[2]\n" \
  "smlalt z22.s, z15.h, z0.h[2]\n" \
  "smlalb z23.s, z15.h, z0.h[3]\n" \
  "smlalt z24.s, z15.h, z0.h[3]\n" \
  "smlalb z25.s, z15.h, z0.h[4]\n" \
  "smlalt z26.s, z15.h, z0.h[4]\n" \
  "smlalb z27.s, z15.h, z0.h[5]\n" \
  "smlalt z28.s, z15.h, z0.h[5]\n" \
  "smlalb z29.s, z15.h, z0.h[6]\n" \
  "smlalt z30.s, z15.h, z0.h[6]\n" \
  "smlalb z6.s,  z15.h, z0.h[7]\n" \
  "smlalt z7.s,  z15.h, z0.h[7]\n" \
  "smlalb z8.s,  z15.h, z4.h[0]\n" \
  "smlalt z9.s,  z15.h, z4.h[0]\n" \
  "ld1b z31.b, p0/Z, [%x[r4]]\n"   \
  "add %x[r4], %x[r4], x10\n"      \
  "sshllb z2.h, z31.b, #0\n"       \
  "sshllt z3.h, z31.b, #0\n"       \
  "ldr q31, [%x[r4]]\n"            \
  "sshllb z5.h, z31.b, #0\n"       \
  "smlalb z19.s, z16.h, z2.h[0]\n" \
  "smlalt z20.s, z16.h, z2.h[0]\n" \
  "smlalb z21.s, z16.h, z2.h[1]\n" \
  "smlalt z22.s, z16.h, z2.h[1]\n" \
  "smlalb z23.s, z16.h, z2.h[2]\n" \
  "smlalt z24.s, z16.h, z2.h[2]\n" \
  "smlalb z25.s, z16.h, z2.h[3]\n" \
  "smlalt z26.s, z16.h, z2.h[3]\n" \
  "smlalb z27.s, z16.h, z2.h[4]\n" \
  "smlalt z28.s, z16.h, z2.h[4]\n" \
  "smlalb z29.s, z16.h, z2.h[5]\n" \
  "smlalt z30.s, z16.h, z2.h[5]\n" \
  "smlalb z6.s,  z16.h, z2.h[6]\n" \
  "smlalt z7.s,  z16.h, z2.h[6]\n" \
  "smlalb z8.s,  z16.h, z2.h[7]\n" \
  "smlalt z9.s,  z16.h, z2.h[7]\n" \
  "smlalb z19.s, z17.h, z3.h[0]\n" \
  "smlalt z20.s, z17.h, z3.h[0]\n" \
  "smlalb z21.s, z17.h, z3.h[1]\n" \
  "smlalt z22.s, z17.h, z3.h[1]\n" \
  "smlalb z23.s, z17.h, z3.h[2]\n" \
  "smlalt z24.s, z17.h, z3.h[2]\n" \
  "smlalb z25.s, z17.h, z3.h[3]\n" \
  "smlalt z26.s, z17.h, z3.h[3]\n" \
  "smlalb z27.s, z17.h, z3.h[4]\n" \
  "smlalt z28.s, z17.h, z3.h[4]\n" \
  "smlalb z29.s, z17.h, z3.h[5]\n" \
  "smlalt z30.s, z17.h, z3.h[5]\n" \
  "smlalb z6.s,  z17.h, z3.h[6]\n" \
  "smlalt z7.s,  z17.h, z3.h[6]\n" \
  "smlalb z8.s,  z17.h, z3.h[7]\n" \
  "smlalt z9.s,  z17.h, z3.h[7]\n" \
  "smlalb z19.s, z18.h, z2.h[1]\n" \
  "smlalt z20.s, z18.h, z2.h[1]\n" \
  "smlalb z21.s, z18.h, z2.h[2]\n" \
  "smlalt z22.s, z18.h, z2.h[2]\n" \
  "smlalb z23.s, z18.h, z2.h[3]\n" \
  "smlalt z24.s, z18.h, z2.h[3]\n" \
  "smlalb z25.s, z18.h, z2.h[4]\n" \
  "smlalt z26.s, z18.h, z2.h[4]\n" \
  "smlalb z27.s, z18.h, z2.h[5]\n" \
  "smlalt z28.s, z18.h, z2.h[5]\n" \
  "smlalb z29.s, z18.h, z2.h[6]\n" \
  "smlalt z30.s, z18.h, z2.h[6]\n" \
  "smlalb z6.s,  z18.h, z2.h[7]\n" \
  "smlalt z7.s,  z18.h, z2.h[7]\n" \
  "smlalb z8.s,  z18.h, z5.h[0]\n" \
  "smlalt z9.s,  z18.h, z5.h[0]\n"

#define COMPUTE_SVE2_2_B4           \
  "ld1b z31.b, p0/Z, [%x[r3]]\n"    \
  "add %x[r3], %x[r3], x10\n"       \
  "sshllb z0.h, z31.b, #0\n"        \
  "sshllt z1.h, z31.b, #0\n"        \
  "ldr q31, [%x[r3]]\n"             \
  "sshllb z4.h, z31.b, #0\n"        \
  "smullb z19.s, z10.h, z2.h[0]\n"  \
  "smullt z20.s, z10.h, z2.h[0]\n"  \
  "smullb z21.s, z10.h, z2.h[1]\n"  \
  "smullt z22.s, z10.h, z2.h[1]\n"  \
  "smullb z23.s, z10.h, z2.h[2]\n"  \
  "smullt z24.s, z10.h, z2.h[2]\n"  \
  "smullb z25.s, z10.h, z2.h[3]\n"  \
  "smullt z26.s, z10.h, z2.h[3]\n"  \
  "smlalb z19.s, z11.h, z3.h[0]\n"  \
  "smlalt z20.s, z11.h, z3.h[0]\n"  \
  "smlalb z21.s, z11.h, z3.h[1]\n"  \
  "smlalt z22.s, z11.h, z3.h[1]\n"  \
  "smlalb z23.s, z11.h, z3.h[2]\n"  \
  "smlalt z24.s, z11.h, z3.h[2]\n"  \
  "smlalb z25.s, z11.h, z3.h[3]\n"  \
  "smlalt z26.s, z11.h, z3.h[3]\n"  \
  "smlalb z19.s, z12.h, z2.h[1]\n"  \
  "smlalt z20.s, z12.h, z2.h[1]\n"  \
  "smlalb z21.s, z12.h, z2.h[2]\n"  \
  "smlalt z22.s, z12.h, z2.h[2]\n"  \
  "smlalb z23.s, z12.h, z2.h[3]\n"  \
  "smlalt z24.s, z12.h, z2.h[3]\n"  \
  "smlalb z25.s,  z12.h, z5.h[0]\n" \
  "smlalt z26.s,  z12.h, z5.h[0]\n" \
  "smlalb z19.s, z13.h, z0.h[0]\n"  \
  "smlalt z20.s, z13.h, z0.h[0]\n"  \
  "smlalb z21.s, z13.h, z0.h[1]\n"  \
  "smlalt z22.s, z13.h, z0.h[1]\n"  \
  "smlalb z23.s, z13.h, z0.h[2]\n"  \
  "smlalt z24.s, z13.h, z0.h[2]\n"  \
  "smlalb z25.s, z13.h, z0.h[3]\n"  \
  "smlalt z26.s, z13.h, z0.h[3]\n"  \
  "smlalb z19.s, z14.h, z1.h[0]\n"  \
  "smlalt z20.s, z14.h, z1.h[0]\n"  \
  "smlalb z21.s, z14.h, z1.h[1]\n"  \
  "smlalt z22.s, z14.h, z1.h[1]\n"  \
  "smlalb z23.s, z14.h, z1.h[2]\n"  \
  "smlalt z24.s, z14.h, z1.h[2]\n"  \
  "smlalb z25.s, z14.h, z1.h[3]\n"  \
  "smlalt z26.s, z14.h, z1.h[3]\n"  \
  "smlalb z19.s, z15.h, z0.h[1]\n"  \
  "smlalt z20.s, z15.h, z0.h[1]\n"  \
  "smlalb z21.s, z15.h, z0.h[2]\n"  \
  "smlalt z22.s, z15.h, z0.h[2]\n"  \
  "smlalb z23.s, z15.h, z0.h[3]\n"  \
  "smlalt z24.s, z15.h, z0.h[3]\n"  \
  "smlalb z25.s,  z15.h, z4.h[0]\n" \
  "smlalt z26.s,  z15.h, z4.h[0]\n" \
  "ld1b z31.b, p0/Z, [%x[r4]]\n"    \
  "add %x[r4], %x[r4], x10\n"       \
  "sshllb z2.h, z31.b, #0\n"        \
  "sshllt z3.h, z31.b, #0\n"        \
  "ldr q31, [%x[r4]]\n"             \
  "sshllb z5.h, z31.b, #0\n"        \
  "smlalb z19.s, z16.h, z2.h[0]\n"  \
  "smlalt z20.s, z16.h, z2.h[0]\n"  \
  "smlalb z21.s, z16.h, z2.h[1]\n"  \
  "smlalt z22.s, z16.h, z2.h[1]\n"  \
  "smlalb z23.s, z16.h, z2.h[2]\n"  \
  "smlalt z24.s, z16.h, z2.h[2]\n"  \
  "smlalb z25.s, z16.h, z2.h[3]\n"  \
  "smlalt z26.s, z16.h, z2.h[3]\n"  \
  "smlalb z19.s, z17.h, z3.h[0]\n"  \
  "smlalt z20.s, z17.h, z3.h[0]\n"  \
  "smlalb z21.s, z17.h, z3.h[1]\n"  \
  "smlalt z22.s, z17.h, z3.h[1]\n"  \
  "smlalb z23.s, z17.h, z3.h[2]\n"  \
  "smlalt z24.s, z17.h, z3.h[2]\n"  \
  "smlalb z25.s, z17.h, z3.h[3]\n"  \
  "smlalt z26.s, z17.h, z3.h[3]\n"  \
  "smlalb z19.s, z18.h, z2.h[1]\n"  \
  "smlalt z20.s, z18.h, z2.h[1]\n"  \
  "smlalb z21.s, z18.h, z2.h[2]\n"  \
  "smlalt z22.s, z18.h, z2.h[2]\n"  \
  "smlalb z23.s, z18.h, z2.h[3]\n"  \
  "smlalt z24.s, z18.h, z2.h[3]\n"  \
  "smlalb z25.s,  z18.h, z5.h[0]\n" \
  "smlalt z26.s,  z18.h, z5.h[0]\n"

#define STORE_LINE1_BLK8                          \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z19.s, p0/M, z19.s, z0.s\n"                \
  "add z20.s, p0/M, z20.s, z1.s\n"                \
  "st2w {z19.s, z20.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z21.s, p0/M, z21.s, z0.s\n"                \
  "add z22.s, p0/M, z22.s, z1.s\n"                \
  "st2w {z21.s, z22.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z23.s, p0/M, z23.s, z0.s\n"                \
  "add z24.s, p0/M, z24.s, z1.s\n"                \
  "st2w {z23.s, z24.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z25.s, p0/M, z25.s, z0.s\n"                \
  "add z26.s, p0/M, z26.s, z1.s\n"                \
  "st2w {z25.s, z26.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z27.s, p0/M, z27.s, z0.s\n"                \
  "add z28.s, p0/M, z28.s, z1.s\n"                \
  "st2w {z27.s, z28.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z29.s, p0/M, z29.s, z0.s\n"                \
  "add z30.s, p0/M, z30.s, z1.s\n"                \
  "st2w {z29.s, z30.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z6.s, p0/M, z6.s, z0.s\n"                  \
  "add z7.s, p0/M, z7.s, z1.s\n"                  \
  "st2w {z6.s, z7.s}, p0, [%x[ptr_out1]]\n"       \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z8.s, p0/M, z8.s, z0.s\n"                  \
  "add z9.s, p0/M, z9.s, z1.s\n"                  \
  "st2w {z8.s, z9.s}, p0, [%x[ptr_out1]]\n"       \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n"

#define STORE_LINE1_BLK4                          \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z19.s, p0/M, z19.s, z0.s\n"                \
  "add z20.s, p0/M, z20.s, z1.s\n"                \
  "st2w {z19.s, z20.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z21.s, p0/M, z21.s, z0.s\n"                \
  "add z22.s, p0/M, z22.s, z1.s\n"                \
  "st2w {z21.s, z22.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z23.s, p0/M, z23.s, z0.s\n"                \
  "add z24.s, p0/M, z24.s, z1.s\n"                \
  "st2w {z23.s, z24.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n" \
  "ld2w {z0.s, z1.s}, p0/Z, [%x[ptr_out1]]\n"     \
  "add z25.s, p0/M, z25.s, z0.s\n"                \
  "add z26.s, p0/M, z26.s, z1.s\n"                \
  "st2w {z25.s, z26.s}, p0, [%x[ptr_out1]]\n"     \
  "add %x[ptr_out1], %x[ptr_out1], x10, lsl #1\n"

template <typename Dtype>
void conv_3x3s2_direct_int8_sve2(const int8_t* din,
                                 Dtype* dout,
                                 int num,
                                 int chout,
                                 int hout,
                                 int wout,
                                 int chin,
                                 int hin,
                                 int win,
                                 const int8_t* weights,
                                 const float* bias,
                                 const operators::ConvParam& param,
                                 Context<TARGET(kARM)>* ctx,
                                 const float* scale) {
  auto paddings = *param.paddings;
  bool flag_bias = param.bias;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = 1.f / act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  const int threads = ctx->threads();
  int llc_size = ctx->llc_size() / 4;
  const int hout_c_block = 8;
  const int hout_r_kernel = 2;
  const int wout_round = ((wout + 3) / 4) * 4;
  const int win_round = wout_round * 2 /*stride_w*/ + 1;

  //! get h block
  //! win_round * chin * hin_r_block * sizeof(int8_t) + wout_round *
  //! hout_c_block * hout_r_block * threads * sizeof(int32_t)= l2_size
  //! win_round = 2 * wout_round + 1
  //! hin_r_block = 2 * hout_r_block + 1
  int hout_r_block =
      (llc_size - 2 * wout_round * chin - chin) /
      ((4 * wout_round + 2) * chin + wout_round * hout_c_block * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block = (hout_r_block / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block * 2 + 1;

  auto tmp_work_space = ctx->workspace_data<int8_t>();
  int zero_size = chout > (win_round + 3) / 4 ? chout : (win_round + 3) / 4;
  int32_t ptr_zero[zero_size];  // NOLINT
  memset(ptr_zero, 0, sizeof(int32_t) * zero_size);
  Dtype ptr_write[wout_round];  // NOLINT

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  //! l2_cache start
  int8_t* pre_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;
  int w_loop_b8 = wout_round / 8;
  int w_loop_b4 = (wout_round - w_loop_b8 * 8) / 4;

  int out_row_stride = hout_c_block * wout_round;
  uint64_t cntb = svcntb();
  uint64_t cntw = svcntw();

  for (int n = 0; n < num; ++n) {
    auto din_batch = din + n * chin * size_in_channel;
    auto dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h * 2 - pad_h;
      int he = hs + h_kernel * 2 + 1;
      prepack_input_nxw(din_batch,
                        pre_din,
                        0,
                        chin,
                        hs,
                        he,
                        ws,
                        we,
                        chin,
                        win,
                        hin,
                        reinterpret_cast<int8_t*>(ptr_zero));

      const int8_t* cblock_inr0 = pre_din;
      const int8_t* cblock_inr1 = cblock_inr0 + in_len;
      const int8_t* cblock_inr2 = cblock_inr1 + in_len;
      const int8_t* cblock_inr3 = cblock_inr2 + in_len;
      const int8_t* cblock_inr4 = cblock_inr3 + in_len;

      LITE_PARALLEL_COMMON_BEGIN(c, tid, chout, 0, hout_c_block) {
#ifdef LITE_USE_THREAD_POOL
        auto pre_out =
            reinterpret_cast<int*>(pre_din + pre_in_size) + tid * pre_out_size;
#elif defined(ARM_WITH_OMP)
        auto pre_out = reinterpret_cast<int*>(pre_din + pre_in_size) +
                       omp_get_thread_num() * pre_out_size;
#else
        auto pre_out = reinterpret_cast<int32_t*>(pre_din + pre_in_size);
#endif
        const int8_t* block_inr0 = cblock_inr0;
        const int8_t* block_inr1 = cblock_inr1;
        const int8_t* block_inr2 = cblock_inr2;
        const int8_t* block_inr3 = cblock_inr3;
        const int8_t* block_inr4 = cblock_inr4;

        const int8_t* weight_c = weights + c * w_stride;
        memset(pre_out, 0, pre_out_size * sizeof(int32_t));
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const int8_t* wc0 = weight_c;
          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;
          const int8_t* inr3 = block_inr3;
          const int8_t* inr4 = block_inr4;
          int32_t* pre_out0 = pre_out + hk * out_row_stride;
          int32_t* pre_out1 = pre_out0 + out_row_stride;
          for (int i = 0; i < chin; ++i) {
            const int8_t* r0 = inr0;
            const int8_t* r1 = inr1;
            const int8_t* r2 = inr2;
            const int8_t* r3 = inr3;
            const int8_t* r4 = inr4;
            int32_t* ptr_out0 = pre_out0;
            int32_t* ptr_out1 = pre_out1;
            int w_loop_8 = w_loop_b8;
            int w_loop_4 = w_loop_b4;

            // clang-format off
            asm volatile(
              INIT_WEIGHT
              "cmp %x[w_loop_8], #0\n"
              "beq 2f\n"
              "1:\n"
              "ptrue p0.b\n"
              "mov x10, %x[cntb]\n"
              COMPUTE_SVE2_1
              STORE_LINE0_BLK8
              COMPUTE_SVE2_2
              STORE_LINE1_BLK8
              "subs %x[w_loop_8], %x[w_loop_8], #1\n"
              "bne 1b\n"
              "2:\n"
              "cmp %x[w_loop_4], #0\n"
              "beq 3f\n"
              "lsr x10, %x[cntb], #1\n"
              "whilelt p0.b, xzr, x10\n"
              COMPUTE_SVE2_1_B4
              "ptrue p0.b\n"
              "mov x10, %x[cntb]\n"
              STORE_LINE0_BLK4
              "lsr x10, %x[cntb], #1\n"
              "whilelt p0.b, xzr, x10\n"
              COMPUTE_SVE2_2_B4
              "ptrue p0.b\n"
              "mov x10, %x[cntb]\n"
              STORE_LINE1_BLK4
              "3:\n"
              : [w_loop_8] "+r"(w_loop_8), [w_loop_4] "+r"(w_loop_4),
              [r0] "+r"(r0), [r1] "+r"(r1),
              [r2] "+r"(r2), [r3] "+r"(r3), [r4] "+r"(r4),
              [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1)
              : [cntb] "r"(cntb), [wc0] "r"(wc0)
              : "cc", "memory", "p0", "x10", "z0","z1","z2","z3","z4","z5",
              "z6","z7","z8","z9","z10","z11","z12","z13","z14",
              "z15","z16","z17","z18","z19","z20","z21","z22",
              "z23","z24","z25","z26","z27","z28","z29","z30","z31"
            );
            // clang-format on
            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
        write_int32_nchwc8_to_nchw(pre_out,
                                   dout_batch,
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
      LITE_PARALLEL_COMMON_END();
    }
  }
}

template void conv_3x3s2_direct_int8_sve2(const int8_t* din,
                                          float* dout,
                                          int num,
                                          int chout,
                                          int hout,
                                          int wout,
                                          int chin,
                                          int hin,
                                          int win,
                                          const int8_t* weights,
                                          const float* bias,
                                          const operators::ConvParam& param,
                                          Context<TARGET(kARM)>* ctx,
                                          const float* scale);

template void conv_3x3s2_direct_int8_sve2(const int8_t* din,
                                          int8_t* dout,
                                          int num,
                                          int chout,
                                          int hout,
                                          int wout,
                                          int chin,
                                          int hin,
                                          int win,
                                          const int8_t* weights,
                                          const float* bias,
                                          const operators::ConvParam& param,
                                          Context<TARGET(kARM)>* ctx,
                                          const float* scale);

#undef INIT_WEIGHT
#undef COMPUTE_SVE2_1
#undef COMPUTE_SVE2_2
#undef COMPUTE_SVE2_1_B4
#undef COMPUTE_SVE2_2_B4
#undef STORE_LINE0_BLK8
#undef STORE_LINE1_BLK8
#undef STORE_LINE0_BLK4
#undef STORE_LINE1_BLK4
#endif

}  // namespace sve2
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
