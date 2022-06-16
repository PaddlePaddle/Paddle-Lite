/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void transpose_0213_buffer(__global const CL_DTYPE* src,
                                    __global CL_DTYPE* dst,
                                    __global const int* out_idxs,
                                    __private const int out_tensor_c,
                                    __private const int out_tensor_h,
                                    __private const int out_tensor_w,
                                    __private const int out_tensor_hw) {
  int hidx = get_global_id(0);       // [0, h) columns of dst src_c
  int widx = get_global_id(1) << 3;  // [0, w) rows of dst
  int chidx = get_global_id(2);      // [0, ch) channels of dst src_h

  // idx = chidx * out_tensor_hw + hidx * out_tensor_w + widx
  const int idx =
      hidx * out_tensor_c * out_tensor_w + chidx * out_tensor_w + widx;
  const int dst_idx =
      chidx * out_tensor_w * out_tensor_h + hidx * out_tensor_w + widx;

  CL_DTYPE8 src_w8 = vload8(0, src + idx);
  // if (hidx == 0 && chidx == 0 && )
  // int8      index4 = vload8(0, out_idxs + idx);
  // if (hidx == 4 && widx == 0 && chidx == 127){
  //   printf("src: %f %f %f %f %f %f %f %f\n", src_w8.s0, src_w8.s1, src_w8.s2,
  //   src_w8.s3, src_w8.s4, src_w8.s5, src_w8.s6, src_w8.s7);
  // }
  vstore8(src_w8, 0, dst + dst_idx);

  // dst[index4.s0] = src_w4.s0;
  // dst[index4.s1] = src_w4.s1;
  // dst[index4.s2] = src_w4.s2;
  // dst[index4.s3] = src_w4.s3;
  // dst[index4.s4] = src_w4.s4;
  // dst[index4.s5] = src_w4.s5;
  // dst[index4.s6] = src_w4.s6;
  // dst[index4.s6] = src_w4.s7;
}

__kernel void transpose_10_buffer(__global const CL_DTYPE* src,
                                  __global CL_DTYPE* dst,
                                  __global const int* out_idxs,
                                  __private const int out_tensor_c,
                                  __private const int out_tensor_h,
                                  __private const int out_tensor_w,
                                  __private const int out_tensor_hw) {
  int hidx = get_global_id(0) << 2;  // [0, h) columns of dst src_c
  int widx = get_global_id(1) << 2;  // [0, w) rows of dst

  if (widx >= out_tensor_w || hidx >= out_tensor_h) {
    return;
  }

  if ((hidx == (out_tensor_h + 3) / 4 - 1) ||
      (widx == (out_tensor_w + 3) / 4 - 1)) {
    for (int i = hidx; i < out_tensor_h; i++) {
      for (int j = widx; i < out_tensor_w; j++) {
        dst[i * out_tensor_w + j] = src[j * out_tensor_h + i];
      }
    }
  } else {
    int src_base_p = widx * out_tensor_h + hidx;
    CL_DTYPE4 src0 = vload4(0, src + src_base_p);
    CL_DTYPE4 src1 = vload4(0, src + src_base_p + out_tensor_h);
    CL_DTYPE4 src2 = vload4(0, src + src_base_p + (out_tensor_h << 1));
    CL_DTYPE4 src3 = vload4(0, src + src_base_p + out_tensor_h * 3);

    CL_DTYPE4 dst0 = (CL_DTYPE4)(src0.s0, src1.s0, src2.s0, src3.s0);
    CL_DTYPE4 dst1 = (CL_DTYPE4)(src0.s1, src1.s1, src2.s1, src3.s1);
    CL_DTYPE4 dst2 = (CL_DTYPE4)(src0.s2, src1.s2, src2.s2, src3.s2);
    CL_DTYPE4 dst3 = (CL_DTYPE4)(src0.s3, src1.s3, src2.s3, src3.s3);

    int dst_base_p = hidx * out_tensor_w + widx;
    vstore4(dst0, 0, dst + dst_base_p);
    vstore4(dst1, 0, dst + dst_base_p + out_tensor_w);
    vstore4(dst2, 0, dst + dst_base_p + (out_tensor_w << 1));
    vstore4(dst3, 0, dst + dst_base_p + out_tensor_w * 3);
  }
}

__kernel void transpose_general_buffer(__global const CL_DTYPE* src,
                                       __global CL_DTYPE* dst,
                                       __global const int* out_idxs,
                                       __private const int out_tensor_c,
                                       __private const int out_tensor_h,
                                       __private const int out_tensor_w,
                                       __private const int out_tensor_hw) {
  int hidx = get_global_id(0);   // [0, h) columns of dst
  int widx = get_global_id(1);   // [0, w) rows of dst
  int chidx = get_global_id(2);  // [0, ch) channels of dst

  // idx = chidx * out_tensor_hw + hidx * out_tensor_w + widx
  const int idx = mad((CL_DTYPE)chidx,
                      (CL_DTYPE)out_tensor_hw,
                      (CL_DTYPE)(mul24(hidx, out_tensor_w) + widx));

  dst[out_idxs[idx]] = src[idx];
  // dst[idx] = src[idx];
}
