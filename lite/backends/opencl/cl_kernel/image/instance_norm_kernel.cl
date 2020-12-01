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

// onnx/pytorch instancenorm by lijian
__kernel void instance_norm_onnx(__private const int in_width,
                                 __private const int in_height,
                                 __private const int in_c_group,
                                 __private const int local_work_size_x,
                                 __private const int local_work_size_y,
                                 __private const float epsilon,
                                 __read_only image2d_t input,
                                 __write_only image2d_t output,
                                 __read_only image2d_t scale,
                                 __read_only image2d_t bias) {
  const int out_cn = get_global_id(0);
  const int n = out_cn / in_c_group;
  const int c = out_cn % in_c_group;
  const int w = get_local_id(1);
  const int h = get_local_id(2);
  const int local_id = w * local_work_size_y + h;
  const int local_total_size = local_work_size_x * local_work_size_y;

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
      sum += read_imagef(input, SAMPLER, (int2)(xOffset + xIndex, yOffset + yIndex));
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
      float4 temp = read_imagef(input, SAMPLER, (int2)(xOffset + xIndex, yOffset + yIndex)) - mean_val;
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
  float4 vscale = read_imagef(scale, sampler, (int2)(c, n*in_c_group));
  float4 vbias  = read_imagef(bias, sampler, (int2)(c, n*in_c_group));
  vscale *= s;

  for (int xIndex = w; xIndex < in_width; xIndex += local_work_size_x) {
    for (int yIndex = h; yIndex < in_height; yIndex += local_work_size_y) {
      int2 intout_pos = (int2)(xOffset + xIndex, yOffset + yIndex);
      float4 in_val = read_imagef(input, SAMPLER, intout_pos);
      half4 out_val = convert_half4((in_val - mean_val) * vscale + vbias);
#ifdef RELU
      out_val = activation(out_val);
#endif
      write_imageh(output, intout_pos, out_val);
    }
  }
}


// paddle instancenorm by zhangxi
__kernel void instance_norm_paddle(__read_only image2d_t input,
                                   __write_only image2d_t output,
                                   __read_only image2d_t scale,
                                   __read_only image2d_t bias,
                                   const float epsilon,
                                   const int in_h,
                                   const int in_w){
    __local CL_DTYPE4 saved_mean[1024];
    __local CL_DTYPE4 saved_variance[1024];

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int gidx = get_group_id(0);
    const int gidy = get_group_id(1);
    const int spatial_size = in_h * in_w;

    CL_DTYPE4 mean = (CL_DTYPE4)(0.f, 0.f, 0.f, 0.f);
    CL_DTYPE4 variance = (CL_DTYPE4)(0.f, 0.f, 0.f, 0.f);
    CL_DTYPE4 vepsilon = (CL_DTYPE4)(epsilon, epsilon, epsilon, epsilon);

    const int x_offset = gidx * in_w;
    const int y_offset = gidy * in_h;

    int2 coor;
    for (int i = lid; i < spatial_size; i += lsize) {
        coor.x = i % in_w + x_offset;
        coor.y = i / in_w + y_offset;
        CL_DTYPE4 pixel = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coor);
        mean += pixel;
        variance += pixel * pixel;
    }
    saved_mean[lid] = mean;
    saved_variance[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    //! do reduction
    int dynamic_size = lsize >> 1;
    for (; dynamic_size > 0; dynamic_size >>= 1){
        if (lid < dynamic_size) {
          saved_mean[lid] += saved_mean[lid + dynamic_size];
          saved_variance[lid] += saved_variance[lid + dynamic_size];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    mean = saved_mean[0] / spatial_size;
    variance = saved_variance[0] / spatial_size - mean * mean;
    variance = rsqrt(variance + vepsilon);
    
    //! do instance norm
    coor.x = gidx;
    coor.y = gidy;
    CL_DTYPE4 vscale = READ_IMG_TYPE(CL_DTYPE_CHAR, scale, SAMPLER, coor);
    vscale *= variance;
    CL_DTYPE4 vbias = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coor);
    for (int i = lid; i < spatial_size; i += lsize) {
        coor.x = i % in_w + x_offset;
        coor.y = i / in_w + y_offset;
        CL_DTYPE4 pixel = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coor);
        pixel = (pixel - mean) * vscale + vbias;
        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, coor, pixel);
    }
}
