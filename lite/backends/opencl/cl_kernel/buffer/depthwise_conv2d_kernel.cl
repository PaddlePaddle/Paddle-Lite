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

__kernel void depthwise_conv2d(const int numel,  // num of elements
                               __global CL_DTYPE* input_data,
                               const int height,
                               const int width,
                               const int conved_channel,
                               const int conved_height,
                               const int conved_width,
                               const int kernel_h,
                               const int kernel_w,
                               const int stride_h,
                               const int stride_w,
                               const int pad_h,
                               const int pad_w,
                               __global CL_DTYPE* output_data,
                               __global CL_DTYPE* weight_data,
                               __global CL_DTYPE* bias_data) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for (index; index < numel; index += tmp) {
    const int pw = index % conved_width;
    const int ph = (index / conved_width) % conved_height;
    const int c = (index / conved_width / conved_height) % conved_channel;
    const int n = index / conved_width / conved_height / conved_channel;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    CL_DTYPE v = 0;
    __global CL_DTYPE* input_slice =
        input_data + (n * conved_channel + c) * height * width;
    __global CL_DTYPE* weight_slice = weight_data + c * kernel_h * kernel_w;
    int khstart = hend < kernel_h ? kernel_h - hend : 0;
    int kwstart = wend < kernel_w ? kernel_w - wend : 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        v += input_slice[h * width + w] *
             weight_slice[(khstart + h - hstart) * kernel_w +
                          (kwstart + w - wstart)];
      }
    }
    if (bias_data != NULL) {
      v += bias_data[c];
    }
#ifdef RELU
    CL_DTYPE alpha;
    output_data[index] = activation(v, alpha);
#else
    output_data[index] = v;
#endif
  }
}
