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
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(model_file, "", "model file path of combined protobuf model");
DEFINE_string(params_file, "", "params file path of combined protobuf model");
DEFINE_string(optimized_model_dir, "", "path of optimized naive buffer model");
DEFINE_string(input_tensor_shape, "1,3,224,224", "shape of input tensors");
DEFINE_string(input_tensor_type, "float32", "data type of input tensors");
DEFINE_string(output_tensor_type, "float32", "data type of output tensors");
DEFINE_string(subgraph_model_cache_dir, "", "dir of subgraph model cache");

namespace paddle {
namespace lite {

// The helper functions for loading and running model from command line and
// verifying output data
std::vector<std::string> TypeParsing(std::string text) {
  return Split(text, ":");
}

std::vector<std::vector<int64_t>> ShapeParsing(std::string text) {
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::string> shape_strings = Split(text, ":");
  shapes.resize(shape_strings.size());
  for (size_t i = 0; i < shape_strings.size(); i++) {
    std::vector<std::string> shape_nums = Split(shape_strings[i], ",");
    for (auto shape_num : shape_nums) {
      shapes[i].push_back(atoi(shape_num.c_str()));
    }
  }
  return shapes;
}

int64_t ShapeProduction(std::vector<int64_t> shape) {
  int64_t s = 1;
  for (int64_t dim : shape) {
    s *= dim;
  }
  return s;
}

void FillInputTensors(
    const std::shared_ptr<lite_api::PaddlePredictor>& predictor,
    const std::vector<std::vector<int64_t>>& input_tensor_shape,
    const std::vector<std::string>& input_tensor_type,
    const float value) {
#define FILL_TENSOR_WITH_TYPE(type)                            \
  auto input_tensor_data = input_tensor->mutable_data<type>(); \
  for (int j = 0; j < input_tensor_size; j++) {                \
    input_tensor_data[j] = static_cast<type>(value);           \
  }
  for (size_t i = 0; i < input_tensor_shape.size(); i++) {
    auto input_tensor = predictor->GetInput(i);
    input_tensor->Resize(input_tensor_shape[i]);
    auto input_tensor_size = ShapeProduction(input_tensor->shape());
    if (input_tensor_type[i] == "float32") {
      FILL_TENSOR_WITH_TYPE(float)
    } else if (input_tensor_type[i] == "int64") {
      FILL_TENSOR_WITH_TYPE(int64_t)
    }
  }
#undef FILL_TENSOR_WITH_TYPE
}

void CheckOutputTensors(
    const std::shared_ptr<lite_api::PaddlePredictor>& tar_predictor,
    const std::shared_ptr<lite_api::PaddlePredictor>& ref_predictor,
    const std::vector<std::string>& output_tensor_type) {
#define CHECK_TENSOR_WITH_TYPE(type)                                          \
  auto tar_output_tensor_data = tar_output_tensor->data<type>();              \
  auto ref_output_tensor_data = ref_output_tensor->data<type>();              \
  for (size_t j = 0; j < ref_output_tensor_size; j++) {                       \
    auto abs_diff =                                                           \
        std::fabs(tar_output_tensor_data[j] - ref_output_tensor_data[j]);     \
    auto rel_diff = abs_diff / (std::fabs(ref_output_tensor_data[j]) + 1e-6); \
    VLOG(5) << "val: " << tar_output_tensor_data[j]                           \
            << " ref: " << ref_output_tensor_data[j]                          \
            << " abs_diff: " << abs_diff << " rel_diff: " << rel_diff;        \
    EXPECT_LT(rel_diff, 0.1);                                                 \
  }
  for (size_t i = 0; i < output_tensor_type.size(); i++) {
    auto tar_output_tensor = tar_predictor->GetOutput(i);
    auto ref_output_tensor = ref_predictor->GetOutput(i);
    auto tar_output_tensor_size = ShapeProduction(tar_output_tensor->shape());
    auto ref_output_tensor_size = ShapeProduction(ref_output_tensor->shape());
    EXPECT_EQ(tar_output_tensor_size, ref_output_tensor_size);
    if (output_tensor_type[i] == "float32") {
      CHECK_TENSOR_WITH_TYPE(float)
    } else if (output_tensor_type[i] == "int64") {
      CHECK_TENSOR_WITH_TYPE(int64_t)
    }
  }
#undef CHECK_TENSOR_WITH_TYPE
}

std::shared_ptr<lite_api::PaddlePredictor> TestModel(
    const std::string& model_dir,
    const std::string& model_file,
    const std::string& params_file,
    const std::vector<lite_api::Place>& valid_places,
    const std::vector<std::vector<int64_t>>& input_tensor_shape,
    const std::vector<std::string>& input_tensor_type,
    const std::string& optimized_model_dir) {
  // Generate optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  cxx_config.set_model_file(model_file);
  cxx_config.set_param_file(params_file);
  cxx_config.set_valid_places(valid_places);
  auto predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(optimized_model_dir,
                                lite_api::LiteModelType::kNaiveBuffer);
  // Load optimized model
  lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(optimized_model_dir + ".nb");
  mobile_config.set_power_mode(lite_api::PowerMode::LITE_POWER_HIGH);
  mobile_config.set_threads(1);
  mobile_config.set_subgraph_model_cache_dir(FLAGS_subgraph_model_cache_dir);
  predictor = lite_api::CreatePaddlePredictor(mobile_config);
  FillInputTensors(predictor, input_tensor_shape, input_tensor_type, 1);
  // Run optimized model
  for (int i = 0; i < FLAGS_warmup; i++) {
    predictor->Run();
  }
  for (int i = 0; i < FLAGS_repeats; i++) {
    FillInputTensors(predictor, input_tensor_shape, input_tensor_type, i);
    auto start = GetCurrentUS();
    predictor->Run();
    LOG(INFO) << i << ", " << GetCurrentUS() - start << "us";
  }
  return predictor;
}

TEST(Subgraph, generate_model_and_check_precision) {
  if (FLAGS_model_dir.empty() && FLAGS_model_file.empty() &&
      FLAGS_params_file.empty()) {
    LOG(INFO) << "Using --model_dir, or --model_file and --params_file to set "
                 "the path of model files.";
    return;
  }
  // Parsing the shape of input tensors from strings, supported formats:
  // "1,3,224,224" and "1,3,224,224:1,80"
  auto input_tensor_shape = ShapeParsing(FLAGS_input_tensor_shape);
  // Parsing the data type of input and output tensors from strings, supported
  // formats: "float32" and "float32:int64:int8"
  auto input_tensor_type = TypeParsing(FLAGS_input_tensor_type);
  auto output_tensor_type = TypeParsing(FLAGS_output_tensor_type);
  std::vector<lite_api::Place> valid_places({
#ifdef LITE_WITH_ARM
      lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_X86
      lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
#endif
  });
  // Generate and run optimized model on CPU as the reference predictor
  auto ref_predictor = TestModel(FLAGS_model_dir,
                                 FLAGS_model_file,
                                 FLAGS_params_file,
                                 valid_places,
                                 input_tensor_shape,
                                 input_tensor_type,
                                 FLAGS_optimized_model_dir + "_ref_opt_model");
// Generate and run optimized model on NPU/XPU as the target predictor
#ifdef LITE_WITH_NPU
  valid_places.push_back(lite_api::Place{TARGET(kNPU), PRECISION(kFloat)});
#endif
#ifdef LITE_WITH_HUAWEI_ASCEND_NPU
  valid_places.push_back(
      lite_api::Place{TARGET(kHuaweiAscendNPU), PRECISION(kFloat)});
#endif
#ifdef LITE_WITH_XTCL
  valid_places.push_back(lite_api::Place{TARGET(kXPU), PRECISION(kFloat)});
#endif
  auto tar_predictor = TestModel(FLAGS_model_dir,
                                 FLAGS_model_file,
                                 FLAGS_params_file,
                                 valid_places,
                                 input_tensor_shape,
                                 input_tensor_type,
                                 FLAGS_optimized_model_dir + "_tar_opt_model");
  // Check the difference of the output tensors between reference predictor and
  // target predictor
  CheckOutputTensors(tar_predictor, ref_predictor, output_tensor_type);
}

}  // namespace lite
}  // namespace paddle
