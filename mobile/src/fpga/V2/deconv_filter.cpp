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

#include "fpga/V2/deconv_filter.h"
#include <memory.h>
#include <algorithm>
// #include "deconv_filter.h"
#include "fpga/V2/filter.h"
// #include "filter.h"
#include "fpga/V2/api.h"

namespace paddle_mobile {
namespace fpga {
namespace deconv_filter {

/*
inverse kernel weights of each channel for every filter
*/
void deconv_inverse_filter(float** data_in, int num, int channel, int width,
                           int height) {
  float* tmp = *data_in;
  int data_size = num * channel * width * height;
  int hw_len = height * width;
  auto tmp_data =
      reinterpret_cast<float*>(fpga_malloc(data_size * sizeof(float)));
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channel; ++j) {
      for (int k = 0; k < hw_len; ++k) {
        tmp_data[i * channel * hw_len + j * hw_len + k] =
            (*data_in)[i * channel * hw_len + j * hw_len + hw_len - k - 1];
      }
    }
  }
  *data_in = tmp_data;
  fpga_free(tmp);
}

/*
    calculate sub padding number
*/
int deconv_calc_sub_pad(int filter_axis, int pad, int stride) {
  if (stride == 0 || ((filter_axis - pad - 1) < 0)) {
    PADDLE_MOBILE_ENFORCE(false, "Wrong deconv parameters");
  }
  return (filter_axis - pad - 1) / stride;
}
int deconv_get_sub_filter_axis(int filter_axis, int stride) {
  return (filter_axis / stride);
}

int deconv_get_sub_out_axis(int image_axis, int sub_pad, int sub_filter_axis) {
  return ((image_axis + 2 * sub_pad - sub_filter_axis) + 1);
}

/*
    (filter_width-pad,filter_width-pad) is the first pixel of sub-pixel image
   position. so the omit rows or columns is (stride - )
*/
int deconv_get_omit(int stride, int filter_width, int pad) {
  PADDLE_MOBILE_ENFORCE(filter_width > pad, "Wrong deconv parameters");
  int idx;
  bool flag = false;
  for (idx = 1; idx <= stride; ++idx) {
    int j = idx;
    for (; j <= filter_width;) {
      if (j == filter_width - pad) {
        flag = true;
        break;
      }
      j = j + stride;
    }
    if (flag) {
      break;
    }
  }

  return (stride - idx);
}

template <typename T>
void deconv_get_sub_filter(T** data_in, int height, int width, int sub_conv_n,
                           int kernel_num, int channel) {
  T* ptr_tmp = *data_in;
  int sub_num = kernel_num * sub_conv_n;
  int sub_h = height / sub_conv_n;
  int sub_w = width / sub_conv_n;

  int sub_filter_size =
      kernel_num * sub_h * sub_w * channel * sub_conv_n * sub_conv_n;

  T* ptr_sub_filter =
      reinterpret_cast<T*>(fpga_malloc(sub_filter_size * sizeof(T)));
  for (int idx = 0; idx < sub_conv_n; ++idx) {
    for (int nn = 0; nn < sub_num; ++nn) {
      int ni = nn % kernel_num;

      int woff = sub_conv_n - 1 - (nn / kernel_num);  //

      for (int hh = 0; hh < sub_h; ++hh) {
        int hi = hh * sub_conv_n + idx % sub_conv_n;
        for (int ww = 0; ww < sub_w; ++ww) {
          int wi = ww * sub_conv_n + woff;  // 1 0

          int sidx = ((nn * sub_h + hh) * sub_w + ww) * channel;   //
          int kidx = ((ni * height + hi) * width + wi) * channel;  //

          fpga_copy(
              ptr_sub_filter + idx * sub_h * sub_w * channel * sub_num + sidx,
              (*data_in) + kidx, channel * sizeof(T));
          // for (int cc =0; cc < channel; ++cc) {
          //     ptr_sub_filter[idx*sub_h*sub_w*channel*sub_num + sidx + cc] =
          //     (*data_in)[kidx + cc];
          // }
        }
      }
    }
  }
  *data_in = ptr_sub_filter;
  fpga_free(ptr_tmp);
}

void deconv_NC_convert(float** filter_in, int kernel_num, int channels,
                       int hw) {
  float* tmp = *filter_in;
  float* ptr_filter = reinterpret_cast<float*>(paddle_mobile::fpga::fpga_malloc(
      hw * kernel_num * channels * sizeof(float)));

  for (int c = 0; c < channels; ++c) {
    for (int n = 0; n < kernel_num; ++n) {
      paddle_mobile::fpga::fpga_copy(ptr_filter + n * hw + kernel_num * hw * c,
                                     tmp + n * channels * hw + c * hw,
                                     hw * sizeof(float));
    }
  }
  *filter_in = ptr_filter;
  paddle_mobile::fpga::fpga_free(tmp);
}

void deconv_format_filter(float** data_in, int num, int channel, int height,
                          int width, int group_num, float max, int stride) {
  int data_size = channel * height * width * num;

  /*{
       float result2 = (float)0;
       string filename = "origin_filter_data";
       api::savefile<float>(filename, (void *)*data_in, data_size, result2);
    }*/

  deconv_inverse_filter(data_in, num, channel, width, height);

  /* {
          float result2 = (float)0;
          string filename = "inverse_filter_data";
          api::savefile<float>(filename, (void *)*data_in, data_size, result2);
   }*/

  filter::quantize(data_in, data_size, max);
  /* {
        char result2 = (char)0;
        string filename = "quantize_filter_data";
        api::savefile<char>(filename, (void *)*data_in, data_size, result2);
 }*/
  char** quantize_data = (char**)data_in;  // NOLINT

  filter::convert_to_hwc(quantize_data, num, channel, height, width);
  /*{
       char result2 = (char)0;
       string filename = "convert_to_hwc_filter_data";
       api::savefile<char>(filename, (void *)*quantize_data, data_size,
  result2);
  }*/

  deconv_get_sub_filter<char>(quantize_data, height, width, stride, num,
                              channel);
  /*{
     char result2 = (char)0;
     string filename = "sub_filter_filter_data";
     api::savefile<char>(filename, (void *)*quantize_data, data_size, result2);
}*/

  int sub_conv_n = stride;
  int sub_h = height / sub_conv_n;
  int sub_w = width / sub_conv_n;
  int sub_chw = sub_h * sub_w * channel;
  int sub_num = sub_conv_n * num;
  int division_capacity = filter::calc_division_capacity(sub_chw);
  int num_per_div_before_alignment =
      filter::calc_num_per_div(sub_num, group_num, division_capacity);
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, FILTER_NUM_ALIGNMENT);
  int div_num = (sub_num + num_per_div_before_alignment - 1) /
                num_per_div_before_alignment;
  int residual = (sub_num) % num_per_div_before_alignment;
  int num_after_alignment = num_per_div_after_alignment *
                                ((residual == 0) ? div_num : (div_num - 1)) +
                            align_to_x(residual, FILTER_NUM_ALIGNMENT);

  char** ptr_ptr_data =
      reinterpret_cast<char**>(fpga_malloc(sub_conv_n * sizeof(char*)));
  int origin_offset = sub_chw * sub_num;
  for (int i = 0; i < sub_conv_n; ++i) {
    (ptr_ptr_data)[i] =
        reinterpret_cast<char*>(fpga_malloc(origin_offset * sizeof(char)));
    fpga_copy((ptr_ptr_data)[i], (*quantize_data) + origin_offset * i,
              origin_offset * sizeof(char));

    /* char result2 = (char)0;
     string filename = "ptr_ptr_data" + to_string(i);
     api::savefile<char>(filename, (void *)(ptr_ptr_data[i]), origin_offset,
     result2);
     */
  }
  // char result2 = (char)0;
  //      string filename = "interleave";
  //      api::savefile<char>(filename, (void *)*ptr_ptr_data, origin_offset,
  //      result2);
  fpga_free(*quantize_data);

  int align_offset =
      align_to_x(sub_chw, FILTER_ELEMENT_ALIGNMENT) * num_after_alignment;
  char* ptr_space = reinterpret_cast<char*>(fpga_malloc(
      sub_conv_n * align_offset * sizeof(char)));  // continuous space
  for (int i = 0; i < sub_conv_n; ++i) {
    char* ptr_tmp = (ptr_ptr_data)[i];

    filter::align_element(&ptr_tmp, sub_num, sub_chw);
    filter::align_num(&ptr_tmp, num_per_div_before_alignment, sub_num, sub_chw);

    filter::reorder(&ptr_tmp, num_after_alignment, sub_chw);
    filter::interleave(&ptr_tmp, num_after_alignment, sub_chw);

    /*   char result2 = (char)0;
       string filename = "interleave" + to_string(i);
       api::savefile<char>(filename, (void *)ptr_tmp, align_offset, result2);
*/
    fpga_copy(ptr_space + i * align_offset, ptr_tmp, align_offset);
    fpga_free(ptr_tmp);
  }
  fpga_free(ptr_ptr_data);
  *data_in = reinterpret_cast<float*>(ptr_space);

  /*    {
        char result2 = (char)0;
         string filename = "ptr_space";
         api::savefile<char>(filename, (void *)ptr_space, sub_conv_n *
     align_offset, result2);
      }*/
  fpga_flush(ptr_space, sub_conv_n * align_offset * sizeof(char));
}

void DWDconv_format_filter(float** data_in, int num, int channel, int height,
                           int width, float* scale_ptr, int stride) {
  deconv_inverse_filter(data_in, num, channel, width, height);

  filter::quantize_to_fp16(data_in, channel, height, width, scale_ptr);
  int16_t** quantize_data = (int16_t**)data_in;  // NOLINT
  filter::convert_to_hwn(quantize_data, channel, height, width);

  deconv_get_sub_filter<int16_t>(quantize_data, height, width, stride, num,
                                 channel);

  filter::align_element_n(quantize_data, channel, height, width);
  fpga_flush(*quantize_data, align_to_x(channel, FILTER_ELEMENT_ALIGNMENT) *
                                 height * width * sizeof(int16_t));
}

}  // namespace deconv_filter
}  // namespace fpga
}  // namespace paddle_mobile
