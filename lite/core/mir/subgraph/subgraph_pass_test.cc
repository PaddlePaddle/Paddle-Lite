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
#include "lite/utils/io.h"
#include "lite/utils/string.h"

DEFINE_string(model_file, "", "model file path of combined protobuf model");
DEFINE_string(params_file, "", "params file path of combined protobuf model");
DEFINE_string(optimized_model_dir, "", "path of optimized naive buffer model");
DEFINE_string(input_tensor_shape, "1,3,224,224", "shape of input tensors");
DEFINE_string(input_tensor_type, "float32", "data type of input tensors");
DEFINE_string(output_tensor_type, "float32", "data type of output tensors");
DEFINE_string(input_file, "", "input data file path");
DEFINE_string(subgraph_model_cache_dir, "", "dir of subgraph model cache");
DEFINE_string(padding, "-1000", "");

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

void ReadInputFromFile(const std::string& input_file,
                       std::vector<std::vector<int64_t>>* inputs) {
  std::vector<std::string> lines = ReadLines(input_file);
  inputs->clear();
  inputs->resize(lines.size());
  for (int i = 0; i < lines.size(); i++) {
    for (auto in : Split(lines[i], " ")) {
      if (!in.empty()) {
        inputs->at(i).push_back(atoi(in.c_str()));
      }
    }
  }
}

void FillTransformerInput(
    const std::shared_ptr<lite_api::PaddlePredictor>& predictor,
    const std::vector<std::vector<int64_t>>& inputs,
    int n) {
  // padding to len=16
  auto input = inputs[n];
  int seq_len = input.size();
  int max_seq_len = 16;
  int n_head = 8;
  int max_out_len = 8;
  // src_word pad value
  int64_t eos_idx = 1;
  // float pad_value = -1e3f;
  float pad_value = atof(FLAGS_padding.c_str());

  // src_word [n,c]  int64
  auto src_word_tensor = predictor->GetInput(0);
  std::vector<int64_t> src_word_dims{1, max_seq_len};
  src_word_tensor->Resize(src_word_dims);
  auto src_word_data = src_word_tensor->mutable_data<int64_t>();
  for (int i = 0; i < seq_len; i++) {
    src_word_data[i] = input[i];
  }
  for (int i = seq_len; i < max_seq_len - seq_len; i++) {
    src_word_data[i] = eos_idx;
  }

  // src_pos  [n,c]  int64  0-len
  auto src_pos_tensor = predictor->GetInput(1);
  std::vector<int64_t> src_pos_dims{1, max_seq_len};
  src_pos_tensor->Resize(src_pos_dims);
  auto src_pos_data = src_pos_tensor->mutable_data<int64_t>();
  for (int i = 0; i < seq_len; i++) {
    src_pos_data[i] = i;
  }
  for (int i = seq_len; i < max_seq_len - seq_len; i++) {
    src_pos_data[i] = 0;
  }

  // src_slf_attn_bias  [n,8,c,c]  float32 0; pad_value: -1e9
  auto src_slf_attn_bias_tensor = predictor->GetInput(2);
  std::vector<int64_t> src_slf_attn_bias_dims{
      1, n_head, max_seq_len, max_seq_len};
  src_slf_attn_bias_tensor->Resize(src_slf_attn_bias_dims);
  auto src_slf_attn_bias_data = src_slf_attn_bias_tensor->mutable_data<float>();
  auto src_slf_attn_bias_size = ShapeProduction(src_slf_attn_bias_dims);
  int offset = 0;
  for (int j = 0; j < src_slf_attn_bias_size / max_seq_len; j++) {
    for (int i = 0; i < seq_len; i++) {
      src_slf_attn_bias_data[offset++] = 0.0f;
    }
    for (int i = seq_len; i < max_seq_len; i++) {
      src_slf_attn_bias_data[offset++] = pad_value;
    }
  }

#if 0
  // trg_word  [2,1]  int64  0; need lod: [[0,2],[0,1,2]]
  auto trg_word_tensor = predictor->GetInput(3);
  std::vector<int64_t> trg_word_dims{2, 1};
  std::vector<std::vector<uint64_t>> trg_word_lod{{0, 2}, {0, 1, 2}};
  trg_word_tensor->Resize(trg_word_dims);
  trg_word_tensor->SetLoD(trg_word_lod);
  auto trg_word_data = trg_word_tensor->mutable_data<int64_t>();
  trg_word_data[0] = 0;
  trg_word_data[1] = 0;

  // init_score  [2,1]  float32  0; need lod: [[0,2],[0,1,2]]
  auto init_score_tensor = predictor->GetInput(4);
  std::vector<int64_t> init_score_dims{2, 1};
  init_score_tensor->Resize(init_score_dims);
  init_score_tensor->SetLoD(trg_word_lod);
  auto init_score_data = init_score_tensor->mutable_data<float>();
  init_score_data[0] = 0.f;
  init_score_data[0] = -1e6;

  // init_idx (2) int32
  auto init_idx_tensor = predictor->GetInput(5);
  std::vector<int64_t> init_idx_dims{2};
  init_idx_tensor->Resize(init_idx_dims);
  auto init_idx_data = init_idx_tensor->mutable_data<int>();
  init_idx_data[0] = 0;
  init_idx_data[1] = 0;

  // trg_slf_attn_bias  [8,8,1,8]  float32
  auto trg_slf_attn_bias_tensor = predictor->GetInput(6);
  std::vector<int64_t> trg_slf_attn_bias_dims{
      max_out_len, n_head, 1, max_out_len};
  trg_slf_attn_bias_tensor->Resize(trg_slf_attn_bias_dims);
  auto trg_slf_attn_bias_data = trg_slf_attn_bias_tensor->mutable_data<float>();
  auto trg_slf_attn_bias_size = ShapeProduction(trg_slf_attn_bias_dims);
  offset = 0;
  for (int k = 0; k < max_out_len; k++) {
    for (int j = 0; j < n_head; j++) {
      for (int i = 0; i < max_out_len; i++) {
        trg_slf_attn_bias_data[offset++] = (i <= k) ? 0.0f : -1e6f;
      }
    }
  }

  // trg_src_attn_bias  [n,8,1,c]  float32
  auto trg_src_attn_bias_tensor = predictor->GetInput(7);
  std::vector<int64_t> trg_src_attn_bias_dims{1, n_head, 1, max_seq_len};
  trg_src_attn_bias_tensor->Resize(trg_src_attn_bias_dims);
  auto trg_src_attn_bias_data = trg_src_attn_bias_tensor->mutable_data<float>();
  auto trg_src_attn_bias_size = ShapeProduction(trg_src_attn_bias_dims);
  offset = 0;
  for (int j = 0; j < trg_src_attn_bias_size / max_seq_len; j++) {
    for (int i = 0; i < seq_len; i++) {
      trg_src_attn_bias_data[offset++] = 0.0f;
    }
    for (int i = seq_len; i < max_seq_len; i++) {
      trg_src_attn_bias_data[offset++] = -1e6f;
    }
  }

  // kv_padding_selection  [8,8,8,1]  float32
  auto kv_padding_selection_tensor = predictor->GetInput(8);
  std::vector<int64_t> kv_padding_selection_dims{
      max_out_len, n_head, max_out_len, 1};
  kv_padding_selection_tensor->Resize(kv_padding_selection_dims);
  auto kv_padding_selection_data =
      kv_padding_selection_tensor->mutable_data<float>();
  auto kv_padding_selection_size = ShapeProduction(kv_padding_selection_dims);
  offset = 0;
  for (int k = 0; k < max_out_len; k++) {
    for (int j = 0; j < n_head; j++) {
      for (int i = 0; i < max_out_len; i++) {
        kv_padding_selection_data[offset++] = (i == k) ? 1.0f : 0.0f;
      }
    }
  }
#endif
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

void SaveOut(std::unique_ptr<const lite_api::Tensor> out, std::string dir) {
  auto out_data = out->data<float>();
  auto out_size = ShapeProduction(out->shape());
  std::vector<std::string> lines;
  for (int i = 0; i < out_size; i++) {
    lines.push_back(std::to_string(out_data[i]));
  }
  WriteLines(lines, dir);
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
  cxx_config.set_subgraph_model_cache_dir(FLAGS_subgraph_model_cache_dir);
  auto predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(optimized_model_dir,
                                lite_api::LiteModelType::kNaiveBuffer);
  // Load optimized model
  lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(optimized_model_dir + ".nb");
  mobile_config.set_power_mode(lite_api::PowerMode::LITE_POWER_HIGH);
  mobile_config.set_threads(1);
  //  mobile_config.set_subgraph_model_cache_dir(FLAGS_subgraph_model_cache_dir);
  predictor = lite_api::CreatePaddlePredictor(mobile_config);
  std::vector<std::vector<int64_t>> inputs{};
  ReadInputFromFile(FLAGS_input_file, &inputs);
  // Run optimized model
  for (int i = 0; i < FLAGS_warmup; i++) {
    FillTransformerInput(predictor, inputs, 0);
    predictor->Run();
  }
  for (int i = 0; i < FLAGS_repeats; i++) {
    // FillInputTensors(predictor, input_tensor_shape, input_tensor_type, i);
    FillTransformerInput(predictor, inputs, 0);
    auto start = GetCurrentUS();
    predictor->Run();
    LOG(INFO) << i << ", " << GetCurrentUS() - start << "us";

#if 0
    auto out_tensor_0 = predictor->GetOutput(0);
    auto out_data_0 = out_tensor_0->data<int64_t>();
    auto out_size_0 = ShapeProduction(out_tensor_0->shape());
    for (int i = 0; i < out_size_0; i++) {
      LOG(INFO) << "-- out_0: " << out_data_0[i];
    }

    auto out_tensor_1 = predictor->GetOutput(1);
    auto out_data_1 = out_tensor_1->data<float>();
    auto out_size_1 = ShapeProduction(out_tensor_1->shape());
    for (int i = 0; i < out_size_1; i++) {
      LOG(INFO) << "-- out_1: " << out_data_1[i];
    }
#else
    for (int j = 0; j < predictor->GetOutputNames().size(); j++) {
      auto out_tensor = predictor->GetOutput(j);
      std::string dir =
          "/data/local/tmp/zpy/out/output_" + std::to_string(j) + ".txt";
      SaveOut(std::move(out_tensor), dir);
    }
#endif
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
      lite_api::Place{TARGET(kARM), PRECISION(kInt64)},
#endif
  });
// Generate and run optimized model on CPU as the reference predictor
#if 0
  auto ref_predictor = TestModel(FLAGS_model_dir,
                                 FLAGS_model_file,
                                 FLAGS_params_file,
                                 valid_places,
                                 input_tensor_shape,
                                 input_tensor_type,
                                 FLAGS_optimized_model_dir + "_ref_opt_model");
#else
// Generate and run optimized model on NPU/XPU as the target predictor
#ifdef LITE_WITH_NPU
  valid_places.push_back(lite_api::Place{TARGET(kNPU), PRECISION(kFloat)});
  valid_places.push_back(lite_api::Place{TARGET(kNPU), PRECISION(kInt64)});
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
//  CheckOutputTensors(tar_predictor, ref_predictor, output_tensor_type);
#endif
}

}  // namespace lite
}  // namespace paddle
