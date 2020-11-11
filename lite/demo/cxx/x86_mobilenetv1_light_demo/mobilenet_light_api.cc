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

#include <gperftools/heap-profiler.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "paddle_api.h"            // NOLINT
using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

std::string ShapePrint(const std::vector<shape_t>& shapes) {
  std::string shapes_str{""};
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    auto shape = shapes[shape_idx];
    std::string shape_str;
    for (auto i : shape) {
      shape_str += std::to_string(i) + ",";
    }
    shapes_str += shape_str;
    shapes_str +=
        (shape_idx != 0 && shape_idx == shapes.size() - 1) ? "" : " : ";
  }
  return shapes_str;
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

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void RunModel(std::string model_dir,
              const std::vector<shape_t>& input_shapes,
              size_t repeats,
              size_t warmup,
              size_t print_output_elem,
              size_t power_mode) {
  // 1. Set MobileConfig
  HeapProfilerStart("MobilenetV1Mem");
  MobileConfig config;
  config.set_model_from_file(model_dir);
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  for (int i = 0; i < 100; i++) {
    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor<MobileConfig>(config);
  }
  sleep(20);
  HeapProfilerDump("initial");

  std::cout << "warmup done\n";

  int i = 1000;
  while (i > 0) {
    // 2. Create PaddlePredictor by MobileConfig
    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor<MobileConfig>(config);
    i--;
  }

  sleep(20);
  HeapProfilerDump("result");
  HeapProfilerStop();
  std::cout << "Create 1000 times done\n";
  std::cin >> name;
}

int main(int argc, char** argv) {
  std::vector<std::string> str_input_shapes;
  std::vector<shape_t> input_shapes{{1, 3, 224, 224}};

  int repeats = 10;
  int warmup = 10;
  int print_output_elem = 0;

  if (argc > 2 && argc < 6) {
    std::cerr << "usage: ./" << argv[0] << "\n"
              << "  <naive_buffer_model_dir>\n"
              << "  <raw_input_shapes>, eg: 1,3,224,224 for 1 input; "
                 "1,3,224,224:1,5 for 2 inputs\n"
              << "  <repeats>\n"
              << "  <warmup>\n"
              << "  <print_output>" << std::endl;
    return 0;
  }

  std::string model_dir = argv[1];
  if (argc >= 6) {
    input_shapes.clear();
    std::string raw_input_shapes = argv[2];
    std::cout << "raw_input_shapes: " << raw_input_shapes << std::endl;
    str_input_shapes = split_string(raw_input_shapes);
    for (size_t i = 0; i < str_input_shapes.size(); ++i) {
      std::cout << "input shape: " << str_input_shapes[i] << std::endl;
      input_shapes.push_back(get_shape(str_input_shapes[i]));
    }

    repeats = atoi(argv[3]);
    warmup = atoi(argv[4]);
    print_output_elem = atoi(argv[5]);
  }
  // set arm power mode:
  // 0 for big cluster, high performance
  // 1 for little cluster
  // 2 for all cores
  // 3 for no bind
  size_t power_mode = 0;

  RunModel(
      model_dir, input_shapes, repeats, warmup, print_output_elem, power_mode);

  return 0;
}
