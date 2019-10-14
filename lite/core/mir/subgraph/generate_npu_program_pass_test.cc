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

#include <gtest/gtest.h>
#include <cmath>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"

DEFINE_string(model_file, "", "model file path of combined protobuf model");
DEFINE_string(params_file, "", "params file path of combined protobuf model");
DEFINE_string(optimized_model_dir, "", "path of optimized naive buffer model");
DEFINE_string(input_tensor_shape, "1,3,224,224", "shapes of input tensors");
DEFINE_int32(output_tensor_num, 1, "number of output tensors");

namespace paddle {
namespace lite {

std::vector<std::vector<int64_t>> ParseShape(std::string txt) {
  std::vector<std::vector<int64_t>> shape;
  while (!txt.empty()) {
    size_t idx = txt.find_first_of(":");
    std::string dims = txt.substr(0, idx);
    std::vector<int64_t> s;
    while (!dims.empty()) {
      size_t idx = dims.find_first_of(",");
      int d = atoi(dims.substr(0, idx).c_str());
      VLOG(3) << d;
      s.push_back(d);
      if (idx == std::string::npos) {
        break;
      } else {
        dims = dims.substr(idx + 1);
      }
    }
    shape.push_back(s);
    if (idx == std::string::npos) {
      break;
    } else {
      txt = txt.substr(idx + 1);
    }
  }
  return shape;
}

int64_t ShapeProduction(std::vector<int64_t> shape) {
  int64_t s = 1;
  for (int64_t dim : shape) {
    s *= dim;
  }
  return s;
}

void FillInputTensor(
    const std::shared_ptr<lite_api::PaddlePredictor>& predictor,
    const std::vector<std::vector<int64_t>>& input_tensor_shape,
    const float value) {
  for (int i = 0; i < input_tensor_shape.size(); i++) {
    auto input_tensor = predictor->GetInput(i);
    input_tensor->Resize(input_tensor_shape[i]);
    auto input_tensor_data = input_tensor->mutable_data<float>();
    auto input_tensor_size = ShapeProduction(input_tensor->shape());
    for (int j = 0; j < input_tensor_size; j++) {
      input_tensor_data[i] = value;
    }
  }
}

void CompareOutputTensor(
    const std::shared_ptr<lite_api::PaddlePredictor>& tar_predictor,
    const std::shared_ptr<lite_api::PaddlePredictor>& ref_predictor,
    const int output_tensor_num) {
  for (int i = 0; i < output_tensor_num; i++) {
    auto tar_output_tensor = tar_predictor->GetOutput(i);
    auto ref_output_tensor = ref_predictor->GetOutput(i);
    auto tar_output_tensor_data = tar_output_tensor->data<float>();
    auto ref_output_tensor_data = ref_output_tensor->data<float>();
    auto tar_output_tensor_size = ShapeProduction(tar_output_tensor->shape());
    auto ref_output_tensor_size = ShapeProduction(ref_output_tensor->shape());
    EXPECT_EQ(tar_output_tensor_size, ref_output_tensor_size);
    for (size_t j = 0; j < ref_output_tensor_size; j++) {
      auto diff =
          std::fabs(tar_output_tensor_data[j] - ref_output_tensor_data[j]) /
          (std::fabs(ref_output_tensor_data[j]) + 1e-6);
      VLOG(3) << diff;
      EXPECT_LT(diff, 0.1);
    }
  }
}

std::shared_ptr<lite_api::PaddlePredictor> TestModel(
    const std::string& model_dir,
    const std::string& model_file,
    const std::string& params_file,
    const lite_api::Place& preferred_place,
    const std::vector<lite_api::Place>& valid_places,
    const std::vector<std::vector<int64_t>>& input_tensor_shape,
    const std::string& optimized_model_dir) {
  // generate optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  cxx_config.set_model_file(model_file);
  cxx_config.set_param_file(params_file);
  cxx_config.set_preferred_place(preferred_place);
  cxx_config.set_valid_places(valid_places);
  auto predictor = lite_api::CreatePaddlePredictor(cxx_config);
  FillInputTensor(predictor, input_tensor_shape, 1);
  predictor->SaveOptimizedModel(optimized_model_dir,
                                lite_api::LiteModelType::kNaiveBuffer);
  // load optimized model
  lite_api::MobileConfig mobile_config;
  mobile_config.set_model_dir(optimized_model_dir);
  mobile_config.set_power_mode(lite_api::PowerMode::LITE_POWER_HIGH);
  mobile_config.set_threads(1);
  predictor = lite_api::CreatePaddlePredictor(mobile_config);
  FillInputTensor(predictor, input_tensor_shape, 1);
  // run optimized model
  for (int i = 0; i < FLAGS_warmup; i++) {
    predictor->Run();
  }
  for (int i = 0; i < FLAGS_repeats; i++) {
    auto start = GetCurrentUS();
    predictor->Run();
    LOG(INFO) << i << ", " << GetCurrentUS() - start << "us";
  }
  return predictor;
}

TEST(NPUSubgraph, compare) {
  // parsing input tensor shape, supported formats: "1,3,224,224"
  // "1,3,224,224:1,80"
  std::vector<std::vector<int64_t>> input_tensor_shape =
      ParseShape(FLAGS_input_tensor_shape);
  // generate and run optimized CPU model
  LOG(INFO) << " ================ CPU ================== ";
  auto cpu_predictor =
      TestModel(FLAGS_model_dir,
                FLAGS_model_file,
                FLAGS_params_file,
                lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
                {lite_api::Place{TARGET(kHost), PRECISION(kFloat)},
                 lite_api::Place{TARGET(kARM), PRECISION(kFloat)}},
                input_tensor_shape,
                FLAGS_optimized_model_dir + "/CPU");
  // generate and run optimized NPU model
  LOG(INFO) << " ================ NPU ================== ";
  auto npu_predictor =
      TestModel(FLAGS_model_dir,
                FLAGS_model_file,
                FLAGS_params_file,
                lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
                {lite_api::Place{TARGET(kHost), PRECISION(kFloat)},
                 lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
                 lite_api::Place{TARGET(kNPU), PRECISION(kFloat)}},
                input_tensor_shape,
                FLAGS_optimized_model_dir + "/NPU");
  // verify results
  CompareOutputTensor(npu_predictor, cpu_predictor, FLAGS_output_tensor_num);
}

}  // namespace lite
}  // namespace paddle
