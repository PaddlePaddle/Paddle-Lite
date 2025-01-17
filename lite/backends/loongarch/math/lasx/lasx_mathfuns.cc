//  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/*
   lasx implementation of sin, cos, sincos, exp and log

   Based on "lite/backends/x86"

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/
#include "lite/backends/loongarch/math/include/mathfuns.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

/* declare some LASX constants -- why can't I figure a better way to do that? */
#define _PS256_CONST(Name, Val)                                   \
  static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = { \
      Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST256(Name, Val)                                  \
  static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = { \
      Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS256_CONST_TYPE(Name, Type, Val)                       \
  static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = { \
      Val, Val, Val, Val, Val, Val, Val, Val}

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1);
_PS256_CONST(cephes_log_q1, -2.12194440e-4);
_PS256_CONST(cephes_log_q2, 0.693359375);

/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
v8sf log256_ps(v8sf x) {
  v8si imm0;
  v8sf one = *(v8sf *)_ps256_1;  // NOLINT

  v8sf invalid_mask = lasx_xvfcmp_sle_s(x, lasx_setzero_f32());

  x = lasx_max_f32(x, *(v8sf *)_ps256_min_norm_pos);  // NOLINT
  /* cut off denormalized stuff */                     // NOLINT

  // can be done with LASX
  imm0 = lasx_srli_i32(lasx_castf32_m256i(x), 23);

  /* keep only the fractional part */
  x = lasx_and_f32(x, *(v8sf *)_ps256_inv_mant_mask);  // NOLINT
  x = lasx_or_f32(x, *(v8sf *)_ps256_0p5);             // NOLINT

  // this is again another LASX instruction
  imm0 = lasx_sub_i32(imm0, *(v8si *)_pi32_256_0x7f);  // NOLINT
  v8sf e = lasx_cvti32_f32(imm0);

  e = lasx_add_f32(e, one);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  v8sf mask =
      lasx_xvfcmp_slt_s(x, *(v8sf *)_ps256_cephes_SQRTHF);  // NOLINT
  v8sf tmp = lasx_and_f32(x, mask);
  x = lasx_sub_f32(x, one);
  e = lasx_sub_f32(e, lasx_and_f32(one, mask));
  x = lasx_add_f32(x, tmp);

  v8sf z = lasx_mul_f32(x, x);

  v8sf y = *(v8sf *)_ps256_cephes_log_p0;  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p1);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p2);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p3);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p4);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p5);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p6);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p7);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_log_p8);  // NOLINT
  y = lasx_mul_f32(y, x);

  y = lasx_mul_f32(y, z);

  tmp = lasx_mul_f32(e, *(v8sf *)_ps256_cephes_log_q1);  // NOLINT
  y = lasx_add_f32(y, tmp);

  tmp = lasx_mul_f32(z, *(v8sf *)_ps256_0p5);  // NOLINT
  y = lasx_sub_f32(y, tmp);

  tmp = lasx_mul_f32(e, *(v8sf *)_ps256_cephes_log_q2);  // NOLINT
  x = lasx_add_f32(x, y);
  x = lasx_add_f32(x, tmp);
  x = lasx_or_f32(x, invalid_mask);  // negative arg will be NAN
  return x;
}

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

v8sf exp256_ps(v8sf x) {
  v8sf tmp = lasx_setzero_f32(), fx;
  v8si imm0;
  v8sf one = *(v8sf *)_ps256_1;  // NOLINT

  x = lasx_min_f32(x, *(v8sf *)_ps256_exp_hi);  // NOLINT
  x = lasx_max_f32(x, *(v8sf *)_ps256_exp_lo);  // NOLINT

  /* express exp(x) as exp(g + n*log(2)) */
  fx = lasx_mul_f32(x, *(v8sf *)_ps256_cephes_LOG2EF);  // NOLINT
  fx = lasx_add_f32(fx, *(v8sf *)_ps256_0p5);           // NOLINT

  // imm0 = lasx_cvttf32_i32(fx);
  // tmp  = lasx_cvti32_f32(imm0);

  tmp = lasx_floor_f32(fx);

  /* if greater, substract 1 */
  v8sf mask = lasx_xvfcmp_slt_s(fx, tmp);
  mask = lasx_and_f32(mask, one);
  fx = lasx_sub_f32(tmp, mask);

  tmp = lasx_mul_f32(fx, *(v8sf *)_ps256_cephes_exp_C1);     // NOLINT
  v8sf z = lasx_mul_f32(fx, *(v8sf *)_ps256_cephes_exp_C2);  // NOLINT
  x = lasx_sub_f32(x, tmp);
  x = lasx_sub_f32(x, z);

  z = lasx_mul_f32(x, x);

  v8sf y = *(v8sf *)_ps256_cephes_exp_p0;  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_exp_p1);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_exp_p2);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_exp_p3);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_exp_p4);  // NOLINT
  y = lasx_mul_f32(y, x);
  y = lasx_add_f32(y, *(v8sf *)_ps256_cephes_exp_p5);  // NOLINT
  y = lasx_mul_f32(y, z);
  y = lasx_add_f32(y, x);
  y = lasx_add_f32(y, one);

  /* build 2^n */
  imm0 = lasx_cvttf32_i32(fx);
  // another two LASX instructions
  imm0 = lasx_add_i32(imm0, *(v8si *)_pi32_256_0x7f);  // NOLINT
  imm0 = lasx_slli_i32(imm0, 23);
  v8sf pow2n = lasx_castm256i_f32(imm0);
  y = lasx_mul_f32(y, pow2n);
  return y;
}

v8sf pow256_ps(v8sf a, v8sf b) {
  // pow(x, m) = exp(m * log(x))
  v8sf vsum = exp256_ps(lasx_mul_f32(b, log256_ps(a)));
  return vsum;
}

_PS256_CONST(minus_cephes_DP1, -0.78515625);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS256_CONST(sincof_p0, -1.9515295891E-4);
_PS256_CONST(sincof_p1, 8.3321608736E-3);
_PS256_CONST(sincof_p2, -1.6666654611E-1);
_PS256_CONST(coscof_p0, 2.443315711809948E-005);
_PS256_CONST(coscof_p1, -1.388731625493765E-003);
_PS256_CONST(coscof_p2, 4.166664568298827E-002);
_PS256_CONST(cephes_FOPI, 1.27323954473516);  // 4 / M_PI

/* evaluation of 8 sines at onces using LASX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
v8sf sin256_ps(v8sf x) {  // any x
  v8sf xmm1, xmm2 = lasx_setzero_f32(), xmm3, sign_bit, y;
  v8si imm0, imm2;

  sign_bit = x;
  /* take the absolute value */
  x = lasx_and_f32(x, *(v8sf *)_ps256_inv_sign_mask);  // NOLINT
  /* extract the sign bit (upper one) */
  sign_bit = lasx_and_f32(sign_bit, *(v8sf *)_ps256_sign_mask);  // NOLINT

  /* scale by 4/Pi */
  y = lasx_mul_f32(x, *(v8sf *)_ps256_cephes_FOPI);  // NOLINT

  /* store the integer part of y in mm0 */
  imm2 = lasx_cvttf32_i32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  // another two LASX instruction
  imm2 = lasx_add_i32(imm2, *(v8si *)_pi32_256_1);     // NOLINT
  imm2 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_inv1);  // NOLINT
  y = lasx_cvti32_f32(imm2);

  /* get the swap sign flag */
  imm0 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_4);  // NOLINT
  imm0 = lasx_slli_i32(imm0, 29);
  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  imm2 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_2);    // NOLINT
  imm2 = lasx_cmpeq_i32(imm2, *(v8si *)_pi32_256_0);  // NOLINT

  v8sf swap_sign_bit = lasx_castm256i_f32(imm0);
  v8sf poly_mask = lasx_castm256i_f32(imm2);
  sign_bit = lasx_xor_f32(sign_bit, swap_sign_bit);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf *)_ps256_minus_cephes_DP1;  // NOLINT
  xmm2 = *(v8sf *)_ps256_minus_cephes_DP2;  // NOLINT
  xmm3 = *(v8sf *)_ps256_minus_cephes_DP3;  // NOLINT
  xmm1 = lasx_mul_f32(y, xmm1);
  xmm2 = lasx_mul_f32(y, xmm2);
  xmm3 = lasx_mul_f32(y, xmm3);
  x = lasx_add_f32(x, xmm1);
  x = lasx_add_f32(x, xmm2);
  x = lasx_add_f32(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v8sf *)_ps256_coscof_p0;  // NOLINT
  v8sf z = lasx_mul_f32(x, x);

  y = lasx_mul_f32(y, z);
  y = lasx_add_f32(y, *(v8sf *)_ps256_coscof_p1);  // NOLINT
  y = lasx_mul_f32(y, z);
  y = lasx_add_f32(y, *(v8sf *)_ps256_coscof_p2);  // NOLINT
  y = lasx_mul_f32(y, z);
  y = lasx_mul_f32(y, z);
  v8sf tmp = lasx_mul_f32(z, *(v8sf *)_ps256_0p5);  // NOLINT
  y = lasx_sub_f32(y, tmp);
  y = lasx_add_f32(y, *(v8sf *)_ps256_1);  // NOLINT

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf *)_ps256_sincof_p0;  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_add_f32(y2, *(v8sf *)_ps256_sincof_p1);  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_add_f32(y2, *(v8sf *)_ps256_sincof_p2);  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_mul_f32(y2, x);
  y2 = lasx_add_f32(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = lasx_and_f32(xmm3, y2);  //, xmm3);
  y = lasx_andnot_f32(xmm3, y);
  y = lasx_add_f32(y, y2);
  /* update the sign */
  y = lasx_xor_f32(y, sign_bit);

  return y;
}

/* almost the same as sin_ps */
v8sf cos256_ps(v8sf x) {  // any x
  v8sf xmm1, xmm2 = lasx_setzero_f32(), xmm3, y;
  v8si imm0, imm2;

  /* take the absolute value */
  x = lasx_and_f32(x, *(v8sf *)_ps256_inv_sign_mask);  // NOLINT

  /* scale by 4/Pi */
  y = lasx_mul_f32(x, *(v8sf *)_ps256_cephes_FOPI);  // NOLINT

  /* store the integer part of y in mm0 */
  imm2 = lasx_cvttf32_i32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = lasx_add_i32(imm2, *(v8si *)_pi32_256_1);     // NOLINT
  imm2 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_inv1);  // NOLINT
  y = lasx_cvti32_f32(imm2);
  imm2 = lasx_sub_i32(imm2, *(v8si *)_pi32_256_2);  // NOLINT

  /* get the swap sign flag */
  imm0 = lasx_andnot_m256i(imm2, *(v8si *)_pi32_256_4);  // NOLINT
  imm0 = lasx_slli_i32(imm0, 29);
  /* get the polynom selection mask */
  imm2 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_2);    // NOLINT
  imm2 = lasx_cmpeq_i32(imm2, *(v8si *)_pi32_256_0);  // NOLINT

  v8sf sign_bit = lasx_castm256i_f32(imm0);
  v8sf poly_mask = lasx_castm256i_f32(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf *)_ps256_minus_cephes_DP1;  // NOLINT
  xmm2 = *(v8sf *)_ps256_minus_cephes_DP2;  // NOLINT
  xmm3 = *(v8sf *)_ps256_minus_cephes_DP3;  // NOLINT
  xmm1 = lasx_mul_f32(y, xmm1);
  xmm2 = lasx_mul_f32(y, xmm2);
  xmm3 = lasx_mul_f32(y, xmm3);
  x = lasx_add_f32(x, xmm1);
  x = lasx_add_f32(x, xmm2);
  x = lasx_add_f32(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v8sf *)_ps256_coscof_p0;  // NOLINT
  v8sf z = lasx_mul_f32(x, x);

  y = lasx_mul_f32(y, z);
  y = lasx_add_f32(y, *(v8sf *)_ps256_coscof_p1);  // NOLINT
  y = lasx_mul_f32(y, z);
  y = lasx_add_f32(y, *(v8sf *)_ps256_coscof_p2);  // NOLINT
  y = lasx_mul_f32(y, z);
  y = lasx_mul_f32(y, z);
  v8sf tmp = lasx_mul_f32(z, *(v8sf *)_ps256_0p5);  // NOLINT
  y = lasx_sub_f32(y, tmp);
  y = lasx_add_f32(y, *(v8sf *)_ps256_1);  // NOLINT

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf *)_ps256_sincof_p0;  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_add_f32(y2, *(v8sf *)_ps256_sincof_p1);  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_add_f32(y2, *(v8sf *)_ps256_sincof_p2);  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_mul_f32(y2, x);
  y2 = lasx_add_f32(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = lasx_and_f32(xmm3, y2);  //, xmm3);
  y = lasx_andnot_f32(xmm3, y);
  y = lasx_add_f32(y, y2);
  /* update the sign */
  y = lasx_xor_f32(y, sign_bit);

  return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could
   replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
void sincos256_ps(v8sf x, v8sf *s, v8sf *c) {
  v8sf xmm1, xmm2, xmm3 = lasx_setzero_f32(), sign_bit_sin, y;
  v8si imm0, imm2, imm4;

  sign_bit_sin = x;
  /* take the absolute value */
  x = lasx_and_f32(x, *(v8sf *)_ps256_inv_sign_mask);  // NOLINT
  /* extract the sign bit (upper one) */
  sign_bit_sin =
      lasx_and_f32(sign_bit_sin, *(v8sf *)_ps256_sign_mask);  // NOLINT

  /* scale by 4/Pi */
  y = lasx_mul_f32(x, *(v8sf *)_ps256_cephes_FOPI);  // NOLINT

  /* store the integer part of y in imm2 */
  imm2 = lasx_cvttf32_i32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = lasx_add_i32(imm2, *(v8si *)_pi32_256_1);     // NOLINT
  imm2 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_inv1);  // NOLINT

  y = lasx_cvti32_f32(imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_4);  // NOLINT
  imm0 = lasx_slli_i32(imm0, 29);
  // v8sf swap_sign_bit_sin = lasx_castm256i_f32(imm0);

  /* get the polynom selection mask for the sine*/
  imm2 = lasx_and_m256i(imm2, *(v8si *)_pi32_256_2);    // NOLINT
  imm2 = lasx_cmpeq_i32(imm2, *(v8si *)_pi32_256_0);  // NOLINT
// v8sf poly_mask = lasx_castm256i_f32(imm2);

  v8sf swap_sign_bit_sin = lasx_castm256i_f32(imm0);
  v8sf poly_mask = lasx_castm256i_f32(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf *)_ps256_minus_cephes_DP1;  // NOLINT
  xmm2 = *(v8sf *)_ps256_minus_cephes_DP2;  // NOLINT
  xmm3 = *(v8sf *)_ps256_minus_cephes_DP3;  // NOLINT
  xmm1 = lasx_mul_f32(y, xmm1);
  xmm2 = lasx_mul_f32(y, xmm2);
  xmm3 = lasx_mul_f32(y, xmm3);
  x = lasx_add_f32(x, xmm1);
  x = lasx_add_f32(x, xmm2);
  x = lasx_add_f32(x, xmm3);

  imm4 = lasx_sub_i32(imm4, *(v8si *)_pi32_256_2);     // NOLINT
  imm4 = lasx_andnot_m256i(imm4, *(v8si *)_pi32_256_4);  // NOLINT
  imm4 = lasx_slli_i32(imm4, 29);

  v8sf sign_bit_cos = lasx_castm256i_f32(imm4);

  sign_bit_sin = lasx_xor_f32(sign_bit_sin, swap_sign_bit_sin);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  v8sf z = lasx_mul_f32(x, x);
  y = *(v8sf *)_ps256_coscof_p0;  // NOLINT

  y = lasx_mul_f32(y, z);
  y = lasx_add_f32(y, *(v8sf *)_ps256_coscof_p1);  // NOLINT
  y = lasx_mul_f32(y, z);
  y = lasx_add_f32(y, *(v8sf *)_ps256_coscof_p2);  // NOLINT
  y = lasx_mul_f32(y, z);
  y = lasx_mul_f32(y, z);
  v8sf tmp = lasx_mul_f32(z, *(v8sf *)_ps256_0p5);  // NOLINT
  y = lasx_sub_f32(y, tmp);
  y = lasx_add_f32(y, *(v8sf *)_ps256_1);  // NOLINT

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf *)_ps256_sincof_p0;  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_add_f32(y2, *(v8sf *)_ps256_sincof_p1);  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_add_f32(y2, *(v8sf *)_ps256_sincof_p2);  // NOLINT
  y2 = lasx_mul_f32(y2, z);
  y2 = lasx_mul_f32(y2, x);
  y2 = lasx_add_f32(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  v8sf ysin2 = lasx_and_f32(xmm3, y2);
  v8sf ysin1 = lasx_andnot_f32(xmm3, y);
  y2 = lasx_sub_f32(y2, ysin2);
  y = lasx_sub_f32(y, ysin1);

  xmm1 = lasx_add_f32(ysin1, ysin2);
  xmm2 = lasx_add_f32(y, y2);

  /* update the sign */
  *s = lasx_xor_f32(xmm1, sign_bit_sin);
  *c = lasx_xor_f32(xmm2, sign_bit_cos);
}

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
