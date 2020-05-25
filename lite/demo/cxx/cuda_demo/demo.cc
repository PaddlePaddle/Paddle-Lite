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

#include <iostream>
#include <vector>
#include "paddle_api.h"  // NOLINT

using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void RunModel(std::string model_dir) {
  // 1. Create CxxConfig
  CxxConfig config;
  config.set_model_file(model_dir + "/__model__");
  config.set_param_file(model_dir + "/__params__");
  config.set_valid_places({
      Place{TARGET(kCUDA), PRECISION(kFloat)},
  });
  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  // 3. Prepare input data
  int num = 1;
  int channels = 3;
  int height = 608;
  int width = 608;
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({num, channels, height, width});
  // fake input data
  std::vector<float> data(num * channels * height * width, 0);
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    data[i] = i % 10 * 0.1;
  }
  input_tensor->CopyFromCpu<float, TargetType::kCUDA>(data.data());
  std::unique_ptr<Tensor> size_tensor(std::move(predictor->GetInput(1)));
  size_tensor->Resize({1, 2});
  std::vector<int> size_data{608, 608};
  size_tensor->CopyFromCpu<int, TargetType::kCUDA>(size_data.data());

  // 4. Run predictor
  predictor->Run();

  // 5. Get output
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  std::vector<float> out_cpu(ShapeProduction(output_tensor->shape()), 0);
  std::cout << "output size is " << ShapeProduction(output_tensor->shape())
            << std::endl;
  output_tensor->CopyToCpu(out_cpu.data());
  for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
    std::cout << "Output[" << i << "]: " << out_cpu[i] << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " model_dir\n";
    exit(1);
  }
  std::string model_dir = argv[1];
  RunModel(model_dir);
  return 0;
}
