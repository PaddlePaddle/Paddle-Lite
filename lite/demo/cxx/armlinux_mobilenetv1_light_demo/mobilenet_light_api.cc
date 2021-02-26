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

#include <sys/time.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

std::string ShapePrint(const shape_t& shape) {
  std::string shape_str{""};
  for (auto i : shape) {
    shape_str += std::to_string(i) + " ";
  }
  return shape_str;
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
              const shape_t& input_shape,
              size_t repeats,
              size_t warmup,
              size_t print_output_elem,
              size_t power_mode) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_dir);
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(
      {input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    data[i] = 1;
  }

  // 4. Run predictor
  for (size_t widx = 0; widx < warmup; ++widx) {
    predictor->Run();
  }

  double sum_duration = 0.0;  // millisecond;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    auto start = GetCurrentUS();

    predictor->Run();

    auto duration = (GetCurrentUS() - start) / 1000.0;
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    std::cout << "run_idx:" << ridx + 1 << " / " << repeats << ": " << duration
              << " ms" << std::endl;
  }
  avg_duration = sum_duration / static_cast<float>(repeats);
  std::cout << "\n======= benchmark summary =======\n"
            << "input_shape(NCHW):" << ShapePrint(input_shape) << "\n"
            << "model_dir:" << model_dir << "\n"
            << "warmup:" << warmup << "\n"
            << "repeats:" << repeats << "\n"
            << "max_duration:" << max_duration << "\n"
            << "min_duration:" << min_duration << "\n"
            << "avg_duration:" << avg_duration << "\n";

  // 5. Get output
  std::cout << "\n====== output summary ====== " << std::endl;
  size_t output_tensor_num = predictor->GetOutputNames().size();
  std::cout << "output tensor num:" << output_tensor_num << std::endl;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor->GetOutput(tidx);
    std::cout << "\n--- output tensor " << tidx << " ---" << std::endl;
    auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, ShapeProduction(out_shape));
    auto out_std_dev = compute_standard_deviation<float>(
        out_data, ShapeProduction(out_shape), true, out_mean);

    std::cout << "output shape(NCHW):" << ShapePrint(out_shape) << std::endl;
    std::cout << "output tensor " << tidx
              << " elem num:" << ShapeProduction(out_shape) << std::endl;
    std::cout << "output tensor " << tidx
              << " standard deviation:" << out_std_dev << std::endl;
    std::cout << "output tensor " << tidx << " mean value:" << out_mean
              << std::endl;

    // print output
    if (print_output_elem) {
      for (int i = 0; i < ShapeProduction(out_shape); ++i) {
        std::cout << "out[" << tidx << "][" << i
                  << "]:" << output_tensor->data<float>()[i] << std::endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  shape_t input_shape{1, 3, 224, 224};  // shape_t ==> std::vector<int64_t>
  int repeats = 10;
  int warmup = 10;
  int print_output_elem = 0;

  if (argc > 2 && argc < 9) {
    std::cerr << "usage: ./" << argv[0] << "\n"
              << "  <naive_buffer_model_dir>\n"
              << "  <input_n>\n"
              << "  <input_c>\n"
              << "  <input_h>\n"
              << "  <input_w>\n"
              << "  <repeats>\n"
              << "  <warmup>\n"
              << "  <print_output>" << std::endl;
    return 0;
  }

  std::string model_dir = argv[1];
  if (argc >= 9) {
    input_shape[0] = atoi(argv[2]);
    input_shape[1] = atoi(argv[3]);
    input_shape[2] = atoi(argv[4]);
    input_shape[3] = atoi(argv[5]);
    repeats = atoi(argv[6]);
    warmup = atoi(argv[7]);
    print_output_elem = atoi(argv[8]);
  }
  // set arm power mode:
  // 0 for big cluster, high performance
  // 1 for little cluster
  // 2 for all cores
  // 3 for no bind
  size_t power_mode = 0;

  RunModel(
      model_dir, input_shape, repeats, warmup, print_output_elem, power_mode);

  return 0;
}
