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
#include "paddle_api.h"            //NOLINT
using namespace paddle::lite_api;  // NOLINT
int g_batch_size = 1;
int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void RunModel(std::string model_dir, int im_height, int im_width) {
  CxxConfig config;
  config.set_model_dir(model_dir);
  // config.set_model_file(model_dir+"/model");
  // config.set_param_file(model_dir+"/params");
  config.set_valid_places({Place{TARGET(kBM), PRECISION(kFloat)},
                           Place{TARGET(kHost), PRECISION(kFloat)}});
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);
  auto cloned_predictor = predictor->Clone();

  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, im_height, im_width});
  auto* data = input_tensor->mutable_data<float>();
  int item_size = ShapeProduction(input_tensor->shape());
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }
  for (int i = 0; i < 100; i++) {
    predictor->Run();
  }

  FILE* fp = fopen("result.txt", "wb");
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  std::cout << "Output shape " << output_tensor->shape()[1] << std::endl;
  for (int i = 0; i < ShapeProduction(output_tensor->shape()); i++) {
    fprintf(fp, "%f\n", output_tensor->data<float>()[i]);
  }

  fclose(fp);

  std::unique_ptr<Tensor> cloned_input_tensor(
      std::move(cloned_predictor->GetInput(0)));
  cloned_input_tensor->Resize({1, 3, im_height, im_width});
  auto* cloned_data = cloned_input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(cloned_input_tensor->shape()); ++i) {
    cloned_data[i] = 1;
  }

  for (int i = 0; i < 1; i++) {
    cloned_predictor->Run();
  }
  std::unique_ptr<const Tensor> cloned_output_tensor(
      std::move(cloned_predictor->GetOutput(0)));
  std::cout << "cloned_Output shape " << cloned_output_tensor->shape()[1]
            << std::endl;
  for (int i = 0; i < ShapeProduction(cloned_output_tensor->shape());
       i += 100) {
    std::cout << "cloned_Output[" << i
              << "]: " << cloned_output_tensor->data<float>()[i] << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " naive_buffer_model_dir\n";
    exit(1);
  }
  std::string model_dir = argv[1];
  int im_height = std::stoi(argv[2]);
  int im_width = std::stoi(argv[3]);
  RunModel(model_dir, im_height, im_width);
  std::cout << "Done" << std::endl;
  return 0;
}
