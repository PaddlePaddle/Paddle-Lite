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

#include <gflags/gflags.h>
#include <iostream>
#include <vector>
#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_full_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT

DEFINE_string(model_dir, "", "Model dir path.");
DEFINE_string(optimized_model_dir, "", "Optimized model dir.");
DEFINE_bool(prefer_int8_kernel, false, "Prefer to run model with int8 kernels");

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

// 0. Enable OpenCL, if needed
// Enable `DEMO_WITH_OPENCL` macro below, if user need use gpu(opencl)
// #define DEMO_WITH_OPENCL
void RunModel() {
  // 1. Set CxxConfig
  CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
#ifdef DEMO_WITH_OPENCL
  std::vector<Place> valid_places{
      Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)},
      Place{TARGET(kARM), PRECISION(kFloat)}};
#else
  std::vector<Place> valid_places{Place{TARGET(kARM), PRECISION(kFloat)}};
#endif
  if (FLAGS_prefer_int8_kernel) {
    valid_places.insert(valid_places.begin(),
                        Place{TARGET(kARM), PRECISION(kInt8)});
  }
  config.set_valid_places(valid_places);

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  // 3. Save the optimized model
  // WARN: The `predictor->SaveOptimizedModel` method must be executed
  // before the `predictor->Run` method. Because some kernels' `PrepareForRun`
  // method maybe change some parameters' values.
  predictor->SaveOptimizedModel(FLAGS_optimized_model_dir,
                                LiteModelType::kNaiveBuffer);

  // 4. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(shape_t({1, 3, 224, 224}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    data[i] = 1;
  }

  // 5. Run predictor
  predictor->Run();

  // 6. Get output
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  std::cout << "Output shape " << output_tensor->shape()[1] << std::endl;
  for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
    std::cout << "Output[" << i << "]: " << output_tensor->data<float>()[i]
              << std::endl;
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "" || FLAGS_optimized_model_dir == "") {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " --model_dir=<your-model-directory>"
              << " --optimized_model_dir=<your-optmized-model-directory> "
              << " --prefer_int8_kernel=[true|false]\n";
    exit(1);
  }
  RunModel();
  return 0;
}
