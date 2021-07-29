/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void initialize_out_as_identity( __global CL_DTYPE* output,
                                          const int height,
                                          const int width, 
                                          __global const CL_DTYPE* input,
                                          int length) {
  const int h = get_global_id(0); // height
  const int w = get_global_id(1); // width
  const int c = get_global_id(2); // channel 

  const int offset = c * height * width + h * width + w;

  __global float* dout_ptr = output + offset;
  *dout_ptr = (h == w) ? 1.f : 0.f;
}

__kernel void partial_gauss_elimination(  __global CL_DTYPE* input,
                                          __global CL_DTYPE* output,
                                          const int height,
                                          const int width ) {
  const int h = get_global_id(0); // height
  const int w = get_global_id(1); // width
  const int c = get_global_id(2); // channel

  __local int pivot_idx[1];
  __local CL_DTYPE pivot_val[1];

  for (int i = 0; i < height; ++i) {
    int glb_off = c * height * width + h * width + w;
    int col_off = c * height * width + h * width + i;
    int row_off = c * height * width + i * width + w;
    int cur_off = c * height * width + i * width + i;
    int swap_i = i;
    if (w == i) {
      // find principal pivot for current iteration
      int cur_val = input[cur_off];
      for (int j = i+1; j < height; ++j) {
        int cmp_off = c * height * width + j * width + i;
        if (fabs((float)input[cmp_off]) > fabs((float)cur_val))
        {
          swap_i = j;
          cur_val = input[cmp_off];
        }
      }
    }
    pivot_idx[0] = swap_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    // swap for the principal pivot row
    if (pivot_idx[0] > i) {
      int pivot_row_off = c * height * width + pivot_idx[0] * width + w;
      CL_DTYPE tmp_in = input[row_off];
      input[row_off] = input[pivot_row_off];
      input[pivot_row_off] = tmp_in;

      CL_DTYPE tmp_out = output[row_off];
      output[row_off] = output[pivot_row_off];
      output[pivot_row_off] = tmp_out;
    }
    pivot_val[0] = input[cur_off];
    barrier(CLK_LOCAL_MEM_FENCE);

    CL_DTYPE factor1 = input[row_off] / pivot_val[0];
    CL_DTYPE factor2 = output[row_off] / pivot_val[0];
    for (int j = 0; j < height; ++j) {
      if (j == i) {
        input[row_off] = input[row_off] / pivot_val[0];
        output[row_off] = output[row_off] / pivot_val[0];
      } else {
        int col_off = c * height * width + j * width + i;
        int glb_off = c * height * width + j * width + w;
        output[glb_off] = output[glb_off] - factor2 * input[col_off];
        input[glb_off] = input[glb_off] - (factor1 * input[col_off]);
      }
    }
  }
}