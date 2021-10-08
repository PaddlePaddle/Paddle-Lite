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

__kernel void matmul_dimc(__read_only image2d_t input_x,
                          __write_only image2d_t output,
                          __global const CL_DTYPE4* input_y,
                          int m,
                          int k,
                          int n) {
  int out_r = get_global_id(1);
  int out_c = get_global_id(0);
  CL_DTYPE4 out0 = 0.f;
  for (int i = 0; i < k; ++i) {
    CL_DTYPE4 in_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_x, SAMPLER, (int2)(i, out_r));
    CL_DTYPE4 in_y = input_y[i * (n / 4) + out_c];
    out0 = mad(in_x.x, in_y, out0);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c, out_r), out0);
}

__kernel void matmul_dim2_LM(__read_only image2d_t input_x,
                             __write_only image2d_t output,
                             __global const CL_DTYPE4* input_y,
                             int M,
                             int K,
                             int N) {
  int out_r = get_global_id(0);
  int out_c = get_global_id(1);

  int row = get_local_id(0);
  int col = get_local_id(1);

  int globalRow = 8 * get_group_id(0) + row;
  int globalCol = 8 * get_group_id(1) + col;

  __local CL_DTYPE4 Asub[8][8];
  __local CL_DTYPE4 Bsub[8][8];

  CL_DTYPE4 out0 = 0.f;
  int numTiles = K / 8;

  for (int t = 0; t < numTiles; t++) {
    if (globalRow < M && globalCol < K) {
      Asub[col][row] = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_x, SAMPLER, (int2)(8 * t + col, globalRow));
    }
    if (globalRow < K && globalCol < N / 4) {
      Bsub[col][row] = input_y[(8 * t + row) * (N / 4) + globalCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (globalRow < M && globalCol < N / 4) {
      for (int k = 0; k < 8; k++) {
        out0 = mad(Asub[k][row].x, Bsub[col][k], out0);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c, out_r), out0);
}

__kernel void matmul_dim2_LM_transY(__read_only image2d_t input_x,
                                    __write_only image2d_t output,
                                    __global const CL_DTYPE* input_y,
                                    int M,
                                    int K,
                                    int N) {
  int out_r = get_global_id(0);
  int out_c = get_global_id(1);

  int row = get_local_id(0);
  int col = get_local_id(1);

  int globalRow = 8 * get_group_id(0) + row;
  int globalCol = 8 * get_group_id(0) + col;

  __local CL_DTYPE4 Asub[32][8];
  __local CL_DTYPE Bsub[32][32];

  CL_DTYPE4 out0 = 0.f;
  int numTiles = K / 32;
  for (int t = 0; t < numTiles; t++) {
    if (globalRow < M && globalCol < K) {
      for (int i = 0; i < 32 / 8; ++i) {
        Asub[col * 4 + i][row] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input_x,
                          SAMPLER,
                          (int2)(32 * t + (col * 4 + i), globalRow));
      }
    }
    if (globalRow < N && globalCol < K) {
      int global_h = 8 * get_group_id(1) + row;
      for (int i = 0; i < 32 / 8; ++i) {
        Bsub[row * 4 + 0][col * 4 + i] =
            input_y[(global_h * 4 + 0) * K + (32 * t + (col * 4 + i))];
        Bsub[row * 4 + 1][col * 4 + i] =
            input_y[(global_h * 4 + 1) * K + (32 * t + (col * 4 + i))];
        Bsub[row * 4 + 2][col * 4 + i] =
            input_y[(global_h * 4 + 2) * K + (32 * t + (col * 4 + i))];
        Bsub[row * 4 + 3][col * 4 + i] =
            input_y[(global_h * 4 + 3) * K + (32 * t + (col * 4 + i))];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (globalRow < M && globalCol < N / 4) {
      for (int k = 0; k < 32; k++) {
        out0.x = mad(Asub[k][row].x, Bsub[col * 4 + 0][k], out0.x);
        out0.y = mad(Asub[k][row].x, Bsub[col * 4 + 1][k], out0.y);
        out0.z = mad(Asub[k][row].x, Bsub[col * 4 + 2][k], out0.z);
        out0.w = mad(Asub[k][row].x, Bsub[col * 4 + 3][k], out0.w);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c, out_r), out0);
}