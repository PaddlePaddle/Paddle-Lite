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

#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"
#include "fpga/api.h"

namespace frame = paddle_mobile::framework;
namespace fpga = paddle_mobile::fpga;
using std::cout;
using std::endl;

void test_format_image() {
  std::vector<int> dims{1, 1, 3, 3};
  std::vector<float> elements{1, 2, 3, 4, 5, 6, 7, 8, 9};
  frame::DDim ddim = frame::make_ddim(dims);
  frame::Tensor image(elements, ddim);
  int num = image.numel();
  float *data_ptr = image.mutable_data<float>();

  for (int i = 0; i < num; i++) {
    cout << data_ptr[i] << " ";
  }
  cout << endl;

  fpga::format_image(&image);
  data_ptr = image.mutable_data<float>();

  for (int i = 0; i < 48; i++) {
    cout << data_ptr[i] << " ";
  }
  cout << endl;
  auto dd = image.dims();
  cout << dims[0] << dims[1] << dims[2] << dims[3] << endl;
}

void test_fill_conv_arg() {
  Tensor input, out, filter;
  DLOG << "Setup input";
  SetupTensor<int16_t>(&input, {1, 250, 32, 30}, static_cast<int16_t>(0),
                       static_cast<int16_t>(1));

  DLOG << "Setup filter";
  SetupTensor<float>(&filter, {1001, 250, 3, 3}, static_cast<float>(0),
                     static_cast<float>(1));

  DLOG << "Setup output";
  SetupTensor<int16_t>(&out, {1, 1001, 32, 30}, static_cast<int16_t>(0),
                       static_cast<int16_t>(1));
  auto bs_ptr = (float *)fpga::fpga_malloc(2 * 1001 * sizeof(float));

  DLOG << "find max";
  float max_value = fpga::filter_find_max(&filter);
  DLOG << "format filter";
  fpga::format_filter(&filter, max_value, 1);

  DLOG << "format bs_ptr";
  int element_num_per_div = fpga::get_filter_num_per_div(&filter, 1);
  fpga::format_bias_scale_array(&bs_ptr, element_num_per_div, 1001);

  DLOG << "format ofm";
  fpga::format_fp16_ofm(&out);
  DLOG << "Build arg";

  fpga::WrapperConvArgs arg;
  fpga::fill_conv_arg(&arg, &input, &out, &filter, true, 1, 1, 1, 1, 1, bs_ptr);
  DLOG << "splitNum: " << arg.split_num << "  group_num:" << arg.group_num
       << "  filter_num:" << arg.filter_num;

  for (int i = 0; i < arg.split_num; i++) {
    DLOG << arg.conv_args[i].filter_num << "   " << arg.conv_args[i].sb_address
         << "   " << arg.conv_args[i].filter_address << "   "
         << arg.conv_args[i].filter_scale_address;
  }
}

int main() {
  test_format_image();
  test_fill_conv_arg();
  return 0;
}
