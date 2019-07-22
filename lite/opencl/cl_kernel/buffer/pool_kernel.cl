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

#define MIN_VALUE -FLT_MAX

__kernel void pool_max(const int numel, // num of elements
                       __global float* input_data,
                       const int num, // num of feature maps
                       const int channels,
                       const int height,
                       const int width,
                       const int pooled_height,
                       const int pooled_width,
                       const int kernel_h,
                       const int kernel_w,
                       const int stride_h,
                       const int stride_w,
                       const int pad_h,
                       const int pad_w,
                       __global float* output_data) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < numel; index += tmp) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = MIN_VALUE;
    int maxidx = -1;
    input_data =
    input_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (input_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = input_data[maxidx];
        }
      }
    }
    output_data[index] = maxval;
  }
}

__kernel void pool_avg(const int numel,
                       __global float* input_data,
                       const int num,
                       const int channels,
                       const int height,
                       const int width,
                       const int pooled_height,
                       const int pooled_width,
                       const int kernel_h,
                       const int kernel_w,
                       const int stride_h,
                       const int stride_w,
                       const int pad_h,
                       const int pad_w,
                       __global float* output_data) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < numel; index+=tmp) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels; int hstart = ph * stride_h - pad_h; int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    float aveval = 0;
    input_data =
    input_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += input_data[h * width + w];
      }
    }
    output_data[index] = aveval / pool_size;
  }
}