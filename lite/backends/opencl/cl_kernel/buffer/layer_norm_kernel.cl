/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void layer_norm_buffer_batch(__global const CL_DTYPE* input,
                                      __global CL_DTYPE* output,
#ifdef SCALE
                                      __global const float* scale,
#endif
#ifdef BIAS
                                      __global const float* bias,
#endif
                                      __private const int batch_size,
                                      __private const int feature_size,
                                      __private const int height,
                                      __private const int channel,
                                      __private const int width,
                                      __private const float epsilon) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_h = out_nh % height;
  int out_n = out_nh / height;
  CL_DTYPE avg = (CL_DTYPE)(0.0f);
  CL_DTYPE var = (CL_DTYPE)(0.0f);
  int index = out_n * channel * height * width;
  int hw = height * width;
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        CL_DTYPE in = input[index + c * hw + h * width + w];
        avg += in;
        var += in * in;
      }
    }
  }
  avg /= feature_size;
  var = var / feature_size - avg * avg;

  var = sqrt(var + CONVERT_TYPE_TO(epsilon, CL_DTYPE));
  int out_index = index + out_c * hw + out_h * width + out_w;
  CL_DTYPE in = input[out_index];
  CL_DTYPE out = (in - avg) / var;
#ifdef SCALE
  out *= (CL_DTYPE)(scale[out_c * hw + out_h * width + out_w]);
#endif
#ifdef BIAS
  out += (CL_DTYPE)(bias[out_c * hw + out_h * width + out_w]);
#endif
  output[out_index] = out;
}

__kernel void layer_norm_buffer_chann(__global const CL_DTYPE* input,
                                      __global CL_DTYPE* output,
#ifdef SCALE
                                      __global const float* scale,
#endif
#ifdef BIAS
                                      __global const float* bias,
#endif
                                      __private const int batch_size,
                                      __private const int feature_size,
                                      __private const int height,
                                      __private const int channel,
                                      __private const int width,
                                      __private const float epsilon) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_h = out_nh % height;
  int out_n = out_nh / height;
  CL_DTYPE avg = (CL_DTYPE)(0.0f);
  CL_DTYPE var = (CL_DTYPE)(0.0f);
  int index = out_n * channel * height * width + out_c * height * width;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      CL_DTYPE in = input[index + h * width + w];
      avg += in;
      var += in * in;
    }
  }
  avg /= feature_size;
  var = var / feature_size - avg * avg;
  var = sqrt(var + CONVERT_TYPE_TO(epsilon, CL_DTYPE));
  int out_index = index + out_h * width + out_w;
  CL_DTYPE in = input[out_index];
  CL_DTYPE out = (in - avg) / var;
#ifdef SCALE
  out *= (CL_DTYPE)(scale[out_h * width + out_w]);
#endif
#ifdef BIAS
  out += (CL_DTYPE)(bias[out_h * width + out_w]);
#endif
  output[out_index] = out;
}

__kernel void layer_norm_buffer_width(__global const CL_DTYPE* input,
                                      __global CL_DTYPE* output,
#ifdef SCALE
                                      __global const float* scale,
#endif
#ifdef BIAS
                                      __global const float* bias,
#endif
                                      __private const int batch_size,
                                      __private const int feature_size,
                                      __private const int height,
                                      __private const int channel,
                                      __private const int width,
                                      __private const float epsilon) {
  const int out_c = get_global_id(0);
  const int g_id_1 = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int local_id = get_local_id(1);
  const int local_work_size = get_local_size(1);

  int out_h = out_nh % height;
  int out_n = out_nh / height;
  CL_DTYPE avg = (CL_DTYPE)(0.0f);
  CL_DTYPE var = (CL_DTYPE)(0.0f);
  int index =
      out_n * channel * height * width + out_c * height * width + out_h * width;
  __global const CL_DTYPE* input_group_p = input + index;
#ifdef LOCAL_MEM_128
  __local CL_DTYPE shared_mem[128];
  __local float shared_mem_1[128];
#elif defined(LOCAL_MEM_64)
  __local CL_DTYPE shared_mem[64];
  __local float shared_mem_1[64];
#else
  __local CL_DTYPE shared_mem[256];
  __local float shared_mem_1[256];
#endif
  CL_DTYPE sum = (CL_DTYPE)(0.0f);
  float sum_2 = (float)(0.0f);
  for (int i = local_id; i < width; i += local_work_size) {
    sum += input_group_p[i];
    sum_2 += convert_float(input_group_p[i]) * convert_float(input_group_p[i]);
  }
  shared_mem[local_id] = sum;
  shared_mem_1[local_id] = sum_2;
  barrier(CLK_LOCAL_MEM_FENCE);

  int reduction_size = local_work_size;
  CL_DTYPE item = (CL_DTYPE)(0.0f);
  CL_DTYPE item_1 = (CL_DTYPE)(0.0f);
  while (reduction_size > 1) {
    int active_thread_limit = reduction_size / 2;
    int offset = (reduction_size + 1) / 2;
    if (local_id < active_thread_limit) {
      shared_mem[local_id] += shared_mem[local_id + offset];
      shared_mem_1[local_id] += shared_mem_1[local_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    reduction_size = offset;
  }
  avg = shared_mem[0];
  var = CONVERT_TYPE_TO(shared_mem_1[0] / feature_size, CL_DTYPE);

  avg /= feature_size;
  var = var - avg * avg;
  var = sqrt(var + CONVERT_TYPE_TO(epsilon, CL_DTYPE));
  int out_index =
      out_n * channel * height * width + out_c * height * width + out_h * width;
  for (int i = local_id; i < width; i += local_work_size) {
    CL_DTYPE in = input[out_index + i];
    CL_DTYPE out = (in - avg) / var;
#ifdef SCALE
    out *= CONVERT_TYPE_TO(scale[i], CL_DTYPE);
#endif
#ifdef BIAS
    out += CONVERT_TYPE_TO(bias[i], CL_DTYPE);
#endif
    output[out_index + i] = out;
  }
}
