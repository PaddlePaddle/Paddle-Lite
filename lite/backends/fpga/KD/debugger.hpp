// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// cpplint off
#pragma once
#include <string>
#include "lite/backends/fpga/lite_tensor.h"
namespace paddle {
namespace lite {

inline void ChwToHwc(Tensor* t, float* dst) {
  int num = t->dims()[0];
  int channel = t->dims()[1];

  int height = 1;
  int width = 1;
  if (t->dims().size() > 2) {
    height = t->dims()[2];
  }
  if (t->dims().size() > 3) {
    width = t->dims()[3];
  }
  // int width = t->dims()[3];
  const float* chw_data = t->data<float>();
  float* hwc_data = dst;

  int chw = channel * height * width;
  int wc = width * channel;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          hwc_data[n * chw + h * wc + w * channel + c] = chw_data[index];
          index++;
        }
      }
    }
  }
}

inline void SaveTensor(lite::Tensor* t,
                       const std::string& name,
                       bool convert = true) {
  float* data = const_cast<float*>(t->data<float>());
  float* dst = new float[t->numel()];
  if (convert) {
    ChwToHwc(t, dst);
    data = dst;
  }

  static int counter = 0;
  std::string file = "arm_" + std::to_string(counter) + name;
  counter++;
  std::ofstream ofs;

  ofs.open(file);

  // float* data = dst;
  for (int i = 0; i < t->numel(); i++) {
    float value = data[i];
    ofs << value << std::endl;
  }
  ofs.close();
  delete[] dst;
}
}  // namespace lite
}  // namespace paddle
// cpplint on
