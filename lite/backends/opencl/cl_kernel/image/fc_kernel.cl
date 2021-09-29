/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>

inline void elt_fuse_func_wrapper(__read_only image2d_t second_input_image,
                                  const int2 pos,
                                  CL_DTYPE4 *value_p) {
  CL_DTYPE4 second_val =
      READ_IMG_TYPE(CL_DTYPE_CHAR, second_input_image, SAMPLER, pos);
  *value_p += second_val;
#ifdef ELT_ACT_FUSE
  *value_p = fmax(*value_p, (CL_DTYPE4)0);
#endif
}

__kernel void fc(__read_only image2d_t input,
                 __write_only image2d_t output,
                 __global CL_COMPUTE_DTYPE16 *weights,
#ifdef BIASE_CH
                 __read_only image2d_t biases,
#endif  // BIASE_CH
#ifdef PRELU
                 __read_only image2d_t prelu_alpha,
#endif  // PRELU
#ifdef ELT_FUSE
                 __read_only image2d_t second_input_image,
#endif  // ELT_FUSE
                 int w,
                 int batch,
                 int in_c_blks,
                 int out_c_blks) {
  int out_n = get_global_id(2);
  int out_c = get_global_id(0);
  int2 tid = (int2)(get_local_id(0), get_local_id(1));
  CL_COMPUTE_DTYPE4 s = (CL_COMPUTE_DTYPE4)(0.0f);
  if (out_n >= batch) return;

  if (out_c < out_c_blks) {
    for (int c = tid.y; c < in_c_blks; c += 4) {
      CL_COMPUTE_DTYPE4 v = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(c, out_n));
      CL_COMPUTE_DTYPE16 w = weights[c * out_c_blks + out_c];
      CL_COMPUTE_DTYPE4 partial = v.x * w.s0123;
      partial += v.y * w.s4567;
      partial += v.z * w.s89ab;
      partial += v.w * w.scdef;
      s += partial;
    }
  }
  __local CL_COMPUTE_DTYPE4 temp[32][4];
  temp[tid.x][tid.y] = s;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (out_c >= out_c_blks) {
    return;
  }
  if (tid.y == 0) {
    s += temp[tid.x][1];
    s += temp[tid.x][2];
    s += temp[tid.x][3];
    int2 output_pos0 = (int2)(out_c, out_n);

#ifdef BIASE_CH
    CL_COMPUTE_DTYPE4 output0 =
        s +
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR, biases, SAMPLER, (int2)(out_c, 0));
#else
    CL_COMPUTE_DTYPE4 output0 = s;
#endif  // BIASE_CH

    CL_COMPUTE_DTYPE4 alpha0;
#ifdef PRELU_CH
    alpha0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
#elif defined(PRELU_ELE)
    alpha0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
#elif defined(PRELU_ALL)
    alpha0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
    alpha0.y = alpha0.x;
    alpha0.z = alpha0.x;
    alpha0.w = alpha0.x;
#endif  // PRELU
    output0 = activation_type4(output0, alpha0);
#ifdef SCALE_ACTIVATION
    output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
#endif

    CL_DTYPE4 out0, out1, out2, out3;
    out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
    out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
    out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
    out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &out0);
#endif
    out1.x = out0.y;
    out2.x = out0.z;
    out3.x = out0.w;

    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c * 4, out_n), out0);
    if (w - out_c * 4 >= 2) {
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c * 4 + 1, out_n), out1);
    }
    if (w - out_c * 4 >= 3) {
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c * 4 + 2, out_n), out2);
    }
    if (w - out_c * 4 >= 4) {
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c * 4 + 3, out_n), out3);
    }
  }
}
