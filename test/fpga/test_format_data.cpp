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

int main() {
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

  return 0;
}
