// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "paddle_api.h"            // NOLINT
using namespace paddle::lite_api;  // NOLINT

// read buffer from file
std::string ReadFile(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  if (!ifile.is_open()) {
    std::cout << "Open file: [" << filename << "] failed.";
  }
  std::ostringstream buf;
  char ch;
  while (buf && ifile.get(ch)) buf.put(ch);
  ifile.close();
  return buf.str();
}

std::string ShapePrint(const shape_t& shape) {
  std::string shape_str{""};
  for (auto i : shape) {
    shape_str += std::to_string(i) + " ";
  }
  return shape_str;
}

std::vector<std::string> split_string(const std::string& str_in) {
  std::vector<std::string> str_out;
  std::string tmp_str = str_in;
  while (!tmp_str.empty()) {
    size_t next_offset = tmp_str.find(":");
    str_out.push_back(tmp_str.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return str_out;
}

std::vector<int64_t> get_shape(const std::string& str_shape) {
  std::vector<int64_t> shape;
  std::string tmp_str = str_shape;
  while (!tmp_str.empty()) {
    int dim = atoi(tmp_str.data());
    shape.push_back(dim);
    size_t next_offset = tmp_str.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return shape;
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

void RunModel(std::string model_dir,
              const std::vector<shape_t>& input_shapes,
              size_t print_output_elem) {
  // 1. Set MobileConfig
  MobileConfig config;
  // Read model buffer from file
  std::string model_buffer = ReadFile(model_dir);

  // Three methods to Load model data from buffer//////////////////////////
  //  Method 1. input is right reference of string, recommended.         //
  //    config.set_model_from_buffer(std::move(model_buffer));           //
  //  Method 2. input is left reference of string                        //
  //    config.set_model_from_buffer(model_buffer);                      //
  //  Method 3. input is char* with length                               //
  //    char* data = model_buffer.data();                                //
  //    int length = model_buffer.length();                              //
  //    config.set_model_from_buffer(data, length);                      //
  ////////////////////////////////////////////////////////////////////////
  config.set_model_from_buffer(std::move(model_buffer));

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data
  for (int j = 0; j < input_shapes.size(); ++j) {
    auto input_tensor = predictor->GetInput(j);
    input_tensor->Resize(input_shapes[j]);
    auto input_data = input_tensor->mutable_data<float>();
    int input_num = 1;
    for (int i = 0; i < input_shapes[j].size(); ++i) {
      input_num *= input_shapes[j][i];
    }

    for (int i = 0; i < input_num; ++i) {
      input_data[i] = 1.f;
    }
  }

  // 4. Run predictor
  predictor->Run();

  // 5. Get output
  std::cout << "\n====== output summary ====== " << std::endl;
  size_t output_tensor_num = predictor->GetOutputNames().size();
  std::cout << "output tensor num:" << output_tensor_num << std::endl;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor->GetOutput(tidx);
    std::cout << "\n--- output tensor " << tidx << " ---" << std::endl;
    // Get shape of this output
    auto out_shape = output_tensor->shape();
    // Compute elements num in this output
    int64_t out_elements_num = 1;
    for (auto i : out_shape) out_elements_num *= i;

    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, out_elements_num);

    std::cout << "output shape(NCHW):" << ShapePrint(out_shape) << std::endl;
    std::cout << "output tensor " << tidx << " elem num:" << out_elements_num
              << std::endl;
    std::cout << "output tensor " << tidx << " mean value:" << out_mean
              << std::endl;

    // print output
    if (print_output_elem) {
      for (int i = 0; i < out_elements_num; ++i) {
        std::cout << "out[" << tidx << "][" << i
                  << "]:" << output_tensor->data<float>()[i] << std::endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  std::vector<std::string> str_input_shapes;
  std::vector<shape_t> input_shapes{
      {1, 3, 224, 224}};  // shape_t ==> std::vector<int64_t>

  int repeats = 10;
  int warmup = 10;
  int print_output_elem = 0;

  if (argc > 2 && argc < 4) {
    std::cerr << "usage: ./" << argv[0] << "\n"
              << "  <path_to_optimized_model>\n"
              << "  <raw_input_shapes>, eg: 1,3,224,224 for 1 input; "
                 "1,3,224,224:1,5 for 2 inputs\n"
              << "  <print_output>" << std::endl;
    return 0;
  }

  std::string model_dir = argv[1];
  if (argc >= 4) {
    input_shapes.clear();
    std::string raw_input_shapes = argv[2];
    std::cout << "raw_input_shapes: " << raw_input_shapes << std::endl;
    str_input_shapes = split_string(raw_input_shapes);
    for (size_t i = 0; i < str_input_shapes.size(); ++i) {
      std::cout << "    input " << i << " shape: " << str_input_shapes[i]
                << std::endl;
      input_shapes.push_back(get_shape(str_input_shapes[i]));
    }

    print_output_elem = atoi(argv[3]);
  }

  RunModel(model_dir, input_shapes, print_output_elem);

  return 0;
}
