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

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "io/paddle_mobile_wrap.h"

int main(int argc, char *argv[]) {
#ifndef PADDLE_MOBILE_FPGA
  paddle_mobile::wrap::Net *net =
      new paddle_mobile::wrap::Net(paddle_mobile::wrap::kGPU_CL);
  net->SetCLPath("/data/local/tmp/bin");
  net->Load("./checked_model/model", "./checked_model/params", false, false, 1,
            true);
  int size = 1 * 3 * 416 * 416;
  std::vector<int64_t> shape{1, 3, 416, 416};
  float *data = new float[size];
  for (int i = 0; i < size; i++) {
    data[i] = 0.0;
  }
  std::ifstream infile;
  infile.open("input.txt");
  for (int i = 0; i < size; i++) {
    infile >> data[i];
  }
  infile.close();
  // input as vector
  // std::vector<float> data_as_vector(data, data + size);
  // auto output = net->Predict(data_as_vector, shape);
  // for (auto item : output) {
  //     std::cout << item << std::endl;
  // }
  // input as float pointer
  paddle_mobile::wrap::Tensor input(data,
                                    paddle_mobile::wrap::make_ddim(shape));
  net->Feed("image", input);
  net->Predict();
  auto output = net->Fetch("save_infer_model/scale_0");
  int output_size = 1;
  std::cout << "output shape: ";
  for (int i = 0; i < output->dims().size(); i++) {
    std::cout << output->dims()[i] << " ";
    output_size *= output->dims()[i];
  }
  std::cout << std::endl;
  std::cout << "output data: ";
  for (int i = 0; i < output_size; i++) {
    std::cout << output->data()[i] << std::endl;
  }
#endif
  return 0;
}
