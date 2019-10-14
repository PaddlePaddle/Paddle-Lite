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

#include "cl_common.h"

__kernel void instancenorm(__private const int in_width,
                        __private const int in_height,
                        __private const int in_c_group,
                        __private const int local_work_size_x,
                        __private const int local_work_size_y,
                        __private const float epsilon,
                        __read_only image2d_t input,
                        __write_only image2d_t output) {
  const int out_cn = get_global_id(0);
  const int n = out_cn / in_c_group;
  const int c = out_cn % in_c_group;
  const int w = get_local_id(1);
  const int h = get_local_id(2);
  const int local_id = w * local_work_size_y + h;
  const int local_total_size = local_work_size_x * local_work_size_y;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#ifdef LOCAL_MEM_128
  __local float4 shared_mem[128];
#elif defined(LOCAL_MEM_64)
  __local float4 shared_mem[64];
#else
  __local float4 shared_mem[256];
#endif
  int xOffset = c * in_width;
  int yOffset = n * in_height;
  float4 sum = 0.0f;
  for (int xIndex = w; xIndex < in_width; xIndex += local_work_size_x) {
    for (int yIndex = h; yIndex < in_height; yIndex += local_work_size_y) {
      sum += read_imagef(input, sampler, (int2)(xOffset + xIndex, yOffset + yIndex));
    }
  }
  shared_mem[local_id] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0.0f;
  if (local_id < 32) {
    for (int i = local_id + 32; i < local_total_size; i += 32) {
      sum += shared_mem[i];
    }
  }
  shared_mem[local_id] += sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0.0f;
  if (local_id == 0) {
    int top = min(32, local_total_size);
    for (int i = 0; i < top; i += 1) {
      sum += shared_mem[i];
    }
    shared_mem[0] = sum / (in_width * in_height);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const float4 mean_val = shared_mem[0];

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0.0f;
  for (int xIndex = w; xIndex < in_width; xIndex += local_work_size_x) {
    for (int yIndex = h; yIndex < in_height; yIndex += local_work_size_y) {
      float4 temp = read_imagef(input, sampler, (int2)(xOffset + xIndex, yOffset + yIndex)) - mean_val;
      sum += temp * temp;
    }
  }
  shared_mem[local_id] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0.0f;
  if (local_id < 32) {
    for (int i = local_id + 32; i < local_total_size; i += 32) {
      sum += shared_mem[i];
    }
  }
  shared_mem[local_id] += sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0.0f;
  if (local_id == 0) {
    int top = min(32, local_total_size);
    for (int i = 0; i < top; i += 1) {
      sum += shared_mem[i];
    }
    shared_mem[0] = sum / (in_width * in_height);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const float4 sigma = sqrt(shared_mem[0] + (float4)(epsilon));

  float4 s = 1 / sigma;

  for (int xIndex = w; xIndex < in_width; xIndex += local_work_size_x) {
    for (int yIndex = h; yIndex < in_height; yIndex += local_work_size_y) {
      int2 intout_pos = (int2)(xOffset + xIndex, yOffset + yIndex);
      float4 in_val = read_imagef(input, sampler, intout_pos);
      half4 out_val = convert_half4((in_val - mean_val) * s);
#ifdef RELU
      out_val = activation(out_val);
#endif
      write_imageh(output, intout_pos, out_val);
    }
  }
}
