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

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "lite/core/program.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

// uncomment line below to print tensors;
// #define FPGA_PRINT_TENSOR

class Debugger {
 public:
  static Debugger& get_instance() {
    static Debugger s_instance;
    return s_instance;
  }

  void registerOutput(std::string op_type, zynqmp::Tensor* tensor) {
    if (op_config[op_type]) {
      tensor->saveToFile(op_type, true);
    }
  }

  void setEnable(bool en) { enabled_ = en; }

 private:
  bool enabled_ = false;

  std::unordered_map<std::string, bool> op_config;
  std::unordered_map<std::string, float> tick_tock_map;
  Debugger() {
    op_config["concat"] = true;
    op_config["pooling"] = true;
    op_config["conv"] = true;
    op_config["dropout"] = true;
    op_config["dwconv"] = true;
    op_config["ew_add"] = true;
    op_config["ew_mul"] = true;
    op_config["crop"] = true;
    op_config["feed"] = true;
    op_config["fetch"] = true;
    op_config["fc"] = true;
    op_config["mul"] = true;
    op_config["boxes"] = true;
    op_config["scores"] = true;
    op_config["nms"] = true;
    op_config["pb_boxes"] = true;
    op_config["pb_variances"] = true;
    op_config["reshape"] = true;
    op_config["softmax"] = true;
    op_config["split"] = true;
  }
};

inline void chw_to_hwc(Tensor* t, float* dst) {
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

inline void read_from_file(lite::Tensor* t, const std::string& path) {
  std::ifstream file_stream;
  file_stream.open(path);
  if (!file_stream) {
    return;
  }
  float* data = t->mutable_data<float>();
  int num = t->numel();
  for (int i = 0; i < num; ++i) {
    float value = 0;
    file_stream >> value;
    data[i] = value;
  }
}

inline void save_float(float* data, const std::string& name, int len) {
  static int counter = 0;
  std::string old_string = paddle::lite::to_string(counter);
  std::string new_string =
      std::string(3 - old_string.length(), '0') + old_string;

  std::string file = "arm_" + new_string + name;
  counter++;

  std::ofstream ofs;
  ofs.open(file);
  for (int i = 0; i < len; i++) {
    float value = data[i];
    ofs << value << std::endl;
  }
  ofs.close();
}

inline void save_tensor(lite::Tensor* t,
                        const std::string& name,
                        bool convert = true) {
  float* data = const_cast<float*>(t->data<float>());
  float* dst = new float[t->numel()];
  if (convert) {
    chw_to_hwc(t, dst);
    data = dst;
  }

  save_float(data, name, t->numel());
  delete[] dst;
}

inline void save_tensor(const lite::Tensor* t,
                        const std::string& name,
                        bool convert = true) {
  float* data = const_cast<float*>(t->data<float>());
  float* dst = new float[t->numel()];
  if (convert) {
    chw_to_hwc(const_cast<lite::Tensor*>(t), dst);
    data = dst;
  }
  save_float(data, name, t->numel());
  delete[] dst;
}
}  // namespace lite
}  // namespace paddle
