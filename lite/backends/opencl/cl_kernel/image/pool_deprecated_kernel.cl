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

__kernel void pool_max(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int in_height,
                       __private const int in_width,
                       __private const int out_height,
                       __private const int out_width,
                       __private const int ksize_h,
                       __private const int ksize_w,
                       __private const int stride_h,
                       __private const int stride_w,
                       __private const int pad_top,
                       __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h, 0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w, 0);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  CL_DTYPE4 max_value = (CL_DTYPE4)(MIN_VALUE);
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      CL_DTYPE4 tmp = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
      max_value = max(max_value, tmp);
    }
  }

  const int pos_out_x = mad24(out_c, out_width, out_w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_out_x, out_nh), max_value);
}

__kernel void pool_avg(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int in_height,
                       __private const int in_width,
                       __private const int out_height,
                       __private const int out_width,
                       __private const int ksize_h,
                       __private const int ksize_w,
                       __private const int stride_h,
                       __private const int stride_w,
                       __private const int pad_top,
                       __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h, 0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w, 0);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  CL_DTYPE4 sum = (CL_DTYPE4)(0.0f);

  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      sum += READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }
  CL_DTYPE div;
#ifdef EXCLUSIVE
  div = (CL_DTYPE)((end_h - start_h) * (end_w - start_w));
#else
  div = (CL_DTYPE)(ksize_w * ksize_h);
#endif
  CL_DTYPE4 avg = sum / div;
  const int pos_out_x = mad24(out_c, out_width, out_w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_out_x, out_nh), avg);
}

__kernel void pool_avg_global(__read_only image2d_t input,
                              __write_only image2d_t output,
                              __private const int in_height,
                              __private const int in_width,
                              __private const int out_height,
                              __private const int out_width,
                              __private const int ksize_h,
                              __private const int ksize_w,
                              __private const int stride_h,
                              __private const int stride_w,
                              __private const int pad_top,
                              __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);   // =1
  const int out_nh = get_global_id(2);  // = n*1

  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  // do not use dtype4 here
  // skip issue for half 2048
  float4 sum = (float4)(0.0f);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  for (int y = 0; y < in_height; ++y) {
    for (int x = 0; x < in_width; ++x) {
      CL_DTYPE4 tmp = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));

      sum = convert_float4(tmp) + sum;
    }
  }
  const float global_size_div = 1.0f / (in_height * in_width);
  CL_DTYPE4 avg;
  avg.x = CONVERT_TYPE_TO((sum.x * global_size_div), CL_COMPUTE_DTYPE);
  avg.y = CONVERT_TYPE_TO((sum.y * global_size_div), CL_COMPUTE_DTYPE);
  avg.z = CONVERT_TYPE_TO((sum.z * global_size_div), CL_COMPUTE_DTYPE);
  avg.w = CONVERT_TYPE_TO((sum.w * global_size_div), CL_COMPUTE_DTYPE);

#ifdef DEBUG
  if (out_c == 0) {
    printf("\033[31msum.x= %f \033 \n  ", sum.x);
    printf("sum.y=%f \n  ", sum.y);
    printf("sum.z=%f \n  ", sum.z);
    printf("sum.w=%f \n  ", sum.w);
    printf("one4.x=%f \n  ", convert_float(one4.x));

    printf("in_height=%d \n  ", in_height);
    printf("in_width=%d \n  ", in_width);
    printf("ksize_h=%d \n  ", ksize_h);
    printf("ksize_w=%d \n  ", ksize_w);
    printf("stride_h=%d \n  ", stride_h);
    printf("stride_w=%d \n  ", stride_w);
    printf("pad_top=%d \n  ", pad_top);
    printf("pad_left=%d \n  ", pad_left);
    printf("out_width=%d \n  ", out_width);
    printf("out_height=%d \n  ", out_height);
    printf("i++=%d \n  ", i++);
    printf("avg.x=%f \n  ", convert_float(avg.x));
    printf("avg.y=%f \n  ", convert_float(avg.y));
    printf("avg.z=%f \n  ", convert_float(avg.z));
    printf("avg.w=%f \n  ", convert_float(avg.w));
  }
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c, out_nh), avg);
}
__kernel void pool_max_global(__read_only image2d_t input,
                              __write_only image2d_t output,
                              __private const int in_height,
                              __private const int in_width,
                              __private const int out_height,
                              __private const int out_width,
                              __private const int ksize_h,
                              __private const int ksize_w,
                              __private const int stride_h,
                              __private const int stride_w,
                              __private const int pad_top,
                              __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);   // =1
  const int out_nh = get_global_id(2);  // = n*1

  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  CL_DTYPE4 max_value = (CL_DTYPE4)(MIN_VALUE);
  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  for (int y = 0; y < in_height; ++y) {
    for (int x = 0; x < in_width; ++x) {
      max_value = max(max_value,
                      READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    input,
                                    SAMPLER,
                                    (int2)(pos_in_x + x, pos_in_y + y)));
    }
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c, out_nh), max_value);
}
