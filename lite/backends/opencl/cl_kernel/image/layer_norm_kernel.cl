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

__kernel void layer_norm_batch(__read_only image2d_t input,
                               __write_only image2d_t output,
#ifdef SCALE
                               __global const float* scale,
#endif
#ifdef BIAS
                               __global const float* bias,
#endif
                               __private const int batch_size,
                               __private const int feature_size,
                               __private const int height,
                               __private const int image_w,
                               __private const int width,
                               __private const float epsilon) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_h = out_nh % height;
  int out_n = out_nh / height;
  CL_DTYPE avg = (CL_DTYPE)(0.0f);
  CL_DTYPE var = (CL_DTYPE)(0.0f);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < image_w; ++w) {
      CL_DTYPE4 in = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(w, out_n * height + h));
      avg += (in.x + in.y + in.z + in.w);
      var += (in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w);
    }
  }
  avg /= feature_size;
  var = var / feature_size - avg * avg;

  var = sqrt(var + CONVERT_TYPE_TO(epsilon, CL_DTYPE));
  int2 pos;
  pos.x = out_c * width + out_w;
  pos.y = out_nh;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, pos);
  CL_DTYPE4 out = (in - avg) / var;
#ifdef SCALE
  int index0 = out_c * 4 * height * width + out_h * width + out_w;
  int index1 = (out_c * 4 + 1) * height * width + out_h * width + out_w;
  int index2 = (out_c * 4 + 2) * height * width + out_h * width + out_w;
  int index3 = (out_c * 4 + 3) * height * width + out_h * width + out_w;
  out *=
      (CL_DTYPE4)(scale[index0], scale[index1], scale[index2], scale[index3]);
#endif
#ifdef BIAS
  int ind0 = out_c * 4 * height * width + out_h * width + out_w;
  int ind1 = (out_c * 4 + 1) * height * width + out_h * width + out_w;
  int ind2 = (out_c * 4 + 2) * height * width + out_h * width + out_w;
  int ind3 = (out_c * 4 + 3) * height * width + out_h * width + out_w;
  out += (CL_DTYPE4)(bias[ind0], bias[ind1], bias[ind2], bias[ind3]);
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, pos, out);
}

__kernel void layer_norm_chann(__read_only image2d_t input,
                               __write_only image2d_t output,
#ifdef SCALE
                               __global const float* scale,
#endif
#ifdef BIAS
                               __global const float* bias,
#endif
                               __private const int batch_size,
                               __private const int feature_size,
                               __private const int height,
                               __private const int image_w,
                               __private const int width,
                               __private const float epsilon) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_h = out_nh % height;
  int out_n = out_nh / height;
  CL_DTYPE4 avg = (CL_DTYPE4)(0.0f);
  CL_DTYPE4 var = (CL_DTYPE4)(0.0f);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      CL_DTYPE4 in =
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input,
                        SAMPLER,
                        (int2)(out_c * width + w, out_n * height + h));
      avg.x += in.x;
      avg.y += in.y;
      avg.z += in.z;
      avg.w += in.w;
      var.x += in.x * in.x;
      var.y += in.y * in.y;
      var.z += in.z * in.z;
      var.w += in.w * in.w;
    }
  }
  avg /= feature_size;
  var = var / feature_size - avg * avg;
  var = sqrt(var + CONVERT_TYPE_TO(epsilon, CL_DTYPE));
  int2 pos;
  pos.x = out_c * width + out_w;
  pos.y = out_nh;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, pos);
  CL_DTYPE4 out = (in - avg) / var;
#ifdef SCALE
  out *= (CL_DTYPE4)(scale[out_h * width + out_w]);
#endif
#ifdef BIAS
  out += (CL_DTYPE4)(bias[out_h * width + out_w]);
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, pos, out);
}

__kernel void layer_norm_width(__read_only image2d_t input,
                               __write_only image2d_t output,
#ifdef SCALE
                               __global const float* scale,
#endif
#ifdef BIAS
                               __global const float* bias,
#endif
                               __private const int batch_size,
                               __private const int feature_size,
                               __private const int height,
                               __private const int image_w,
                               __private const int width,
                               __private const float epsilon) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_h = out_nh % height;
  int out_n = out_nh / height;
  CL_DTYPE4 avg = (CL_DTYPE4)(0.0f);
  CL_DTYPE4 var = (CL_DTYPE4)(0.0f);
  for (int w = 0; w < width; ++w) {
    CL_DTYPE4 in = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(out_c * width + w, out_nh));
    avg.x += in.x;
    avg.y += in.y;
    avg.z += in.z;
    avg.w += in.w;
    var.x += in.x * in.x;
    var.y += in.y * in.y;
    var.z += in.z * in.z;
    var.w += in.w * in.w;
  }
  avg /= feature_size;
  var = var / feature_size - avg * avg;
  var = sqrt(var + CONVERT_TYPE_TO(epsilon, CL_DTYPE));
  int2 pos;
  pos.x = out_c * width + out_w;
  pos.y = out_nh;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, pos);
  CL_DTYPE4 out = (in - avg) / var;
#ifdef SCALE
  out *= (CL_DTYPE4)(scale[out_w]);
#endif
#ifdef BIAS
  out += (CL_DTYPE4)(bias[out_w]);
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, pos, out);
}
