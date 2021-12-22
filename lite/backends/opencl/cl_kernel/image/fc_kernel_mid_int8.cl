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

__kernel void fc_mid_int8(__read_only image2d_t input,
                          __write_only image2d_t output,
                          __global char16 *weights,
#ifdef BIASE_CH
                          __read_only image2d_t biases,
#endif  // BIASE_CH
#ifdef PRELU
                          __read_only image2d_t prelu_alpha,
#endif  // PRELU
#ifdef ELT_FUSE
                          __read_only image2d_t second_input_image,
#endif  // ELT_FUSE
                          int batch,
                          int in_c_blks,
                          int out_c_blks,
                          float scale,
                          float scale_de) {
  int out_n = get_global_id(2);
  int out_c = get_global_id(0);
  int2 tid = (int2)(get_local_id(0), get_local_id(1));
  int4 s = (int4)(0);
  float4 scale_v4 = (float4)scale;
  float4 scale_de_v4 = (float4)scale_de;

  if (out_n >= batch) return;

  if (out_c < out_c_blks) {
    for (int c = tid.y; c < in_c_blks; c += 4) {
      half4 v_h4 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(c, out_n));

      int4 v = convert_int4(v_h4 * convert_half4(scale_v4));

      //   if (out_n == 0 && out_c == 0 && c == 0){
      //       printf("scale: %f\n", scale);
      //       //printf("scale_de: %h\n", scale_de);
      //     //   printf("v_h4: %f %f %f %f\n", v_h4.x, v_h4.y, v_h4.z, v_h4.w);
      //       printf("vvv: %d %d %d %d\n", v.x, v.y, v.z, v.w);
      //   }
      int16 w = convert_int16(weights[c * out_c_blks + out_c]);
      int4 partial = v.x * w.s0123;
      partial += v.y * w.s4567;
      partial += v.z * w.s89ab;
      partial += v.w * w.scdef;
      s += partial;
    }
  }
  __local int4 temp[32][4];
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

    half4 output0 = convert_half4(convert_float4(s) * scale_de_v4);

#ifdef BIASE_CH
    half4 out0 =
        output0 +
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR, biases, SAMPLER, (int2)(out_c, 0));
#else
    half4 out0 = output0;
#endif  // BIASE_CH

    // if (out_n == 0 && out_c == 0){
    //     printf("output0: %d %d %d %d\n", output0.x, output0.y, output0.z,
    //     output0.w);
    // }

    half4 alpha0;
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
    out0 = activation_type4(out0, alpha0);
#ifdef SCALE_ACTIVATION
    out0 = fuse_scale(out0, 1.f, 0.f, 0.f);
#endif

// int4 out0;
// out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
// out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
// out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
// out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

// if (out_n == 0 && out_c == 0){
//     printf("out011: %f\n", convert_float(convert_half(output0.x)));
//     printf("out0: %f\n", convert_float(out0.x));
// }

#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &out0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos0, out0);
  }
}