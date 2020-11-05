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

#include <iostream>
#include <vector>
#include "paddle_api.h"  // NOLINT

using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

int REPEAT_COUNT = 1;
uint32_t BATCH_SIZE = 1;
const PowerMode CPU_POWER_MODE = PowerMode::LITE_POWER_HIGH;
std::vector<float> INPUT_MEAN = {124, 117, 104};
std::vector<float> INPUT_STD = {59, 57, 57};

bool use_first_conv = false;

int main(int argc, char** argv) {
  std::string model_dir = "";
  if (argc < 3) {
    std::cout << "USAGE: ./" << argv[0] << " batch_size  model_path"
              << std::endl;
    return 1;
  } else {
    BATCH_SIZE = std::atoi(argv[1]);
    if (BATCH_SIZE < 1) {
      std::cerr << "invalid batch size" << std::endl;
      return -1;
    }
    model_dir = argv[2];
  }

  // Set MobileConfig
  CxxConfig config;
  config.set_model_dir(model_dir);
  std::vector<Place> valid_places{
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)}
      // Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)}
  };
  config.set_valid_places(valid_places);
  if (use_first_conv) {
    std::vector<float> mean_vec = INPUT_MEAN;
    std::vector<float> std_vec = INPUT_STD;
    config.set_mlu_firstconv_param(mean_vec, std_vec);
  }

  config.set_mlu_core_version(MLUCoreVersion::MLU_270);
  config.set_mlu_core_number(16);
  config.set_mlu_input_layout(DATALAYOUT(kNHWC));

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, 224, 224});
  auto* data = input_tensor->mutable_data<float>();
  int item_size = ShapeProduction(input_tensor->shape());
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }
  for (int i = 0; i < REPEAT_COUNT; i++) {
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
  return 0;
}
