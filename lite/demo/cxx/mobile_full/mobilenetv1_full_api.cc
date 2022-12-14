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
#include "paddle_api.h"         // NOLINT
#include "paddle_use_passes.h"  // NOLINT

/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_full_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
#if defined(_WIN32)
#include "paddle_use_kernels.h"  // NOLINT
#include "paddle_use_ops.h"      // NOLINT
#endif

using namespace paddle::lite_api;  // NOLINT

DEFINE_string(model_dir,
              "",
              "Model dir path. Set it when the model is uncombined format.");
DEFINE_string(model_file,
              "",
              "Model file. Set it when the model is combined format.");
DEFINE_string(param_file,
              "",
              "Param file. Set it when the model is combined format.");
DEFINE_string(optimized_model_dir, "", "Optimized model dir.");
DEFINE_bool(prefer_int8_kernel,
            false,
            "Prefer to run model with int8 kernels.");
DEFINE_int32(power_mode,
             3,
             "power mode: "
             "0 for POWER_HIGH;"
             "1 for POWER_LOW;"
             "2 for POWER_FULL;"
             "3 for NO_BIND");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 10, "warmup times");
DEFINE_int32(repeats, 100, "repeats times");
DEFINE_bool(use_gpu, false, "use opencl backend");

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void RunModel() {
  // 1. Set CxxConfig
  CxxConfig config;
  if (!FLAGS_model_file.empty() && !FLAGS_param_file.empty()) {
    config.set_model_file(FLAGS_model_file);
    config.set_param_file(FLAGS_param_file);
  } else if (!FLAGS_model_dir.empty()) {
    config.set_model_dir(FLAGS_model_dir);
  }
  config.set_power_mode((paddle::lite_api::PowerMode)FLAGS_power_mode);
  config.set_threads(FLAGS_threads);

  std::vector<Place> valid_places;
  if (FLAGS_use_gpu) {
    valid_places.emplace_back(
        Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)});
    valid_places.emplace_back(
        Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder)});
    valid_places.emplace_back(
        Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)});
    valid_places.emplace_back(
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)});
    valid_places.emplace_back(
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageFolder)});
    valid_places.emplace_back(
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)});
    valid_places.emplace_back(
        Place{TARGET(kOpenCL), PRECISION(kInt32), DATALAYOUT(kNCHW)});
    valid_places.emplace_back(Place{TARGET(kARM)});
  } else {
    valid_places.emplace_back(Place{TARGET(kARM), PRECISION(kFloat)});
  }

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
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  for (int j = 0; j < FLAGS_repeats; ++j) {
    predictor->Run();
  }

  // 6. Get output
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  std::cout << "Output shape " << output_tensor->shape()[1] << std::endl;
  for (int i = 0; i < ShapeProduction(output_tensor->shape()); i++) {
    std::cout << "Output[" << i << "]: " << output_tensor->data<float>()[i]
              << std::endl;
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty() &&
          (FLAGS_model_file.empty() || FLAGS_param_file.empty()) ||
      FLAGS_optimized_model_dir.empty()) {
    std::cerr
        << "[ERROR] usage: \n"
        << argv[0] << " --model_dir=                 string  Path to "
                      "PaddlePaddle uncombined model.\n"
        << " --model_file=                string  Model file in PaddlePaddle "
           "combined model.\n"
        << " --param_file=                string  Param file in PaddlePaddle "
           "combined model.\n"
        << " --optimized_model_dir=       string  Path to optmized model "
           "file.\n"
        << " --prefer_int8_kernel=false   bool    Prefer to run model with "
           "int8 kernels.\n"
        << " --power_mode=3               int32   Core binding mode.\n"
        << " --threads=1                  int32   Number of threads.\n"
        << " --warmup=10                  int32   Number of warmups.\n"
        << " --repeats=100                int32   Number of repeats.\n"
        << " --use_gpu=false              bool    Use gpu or not.\n";
    exit(1);
  }

  RunModel();
  return 0;
}
