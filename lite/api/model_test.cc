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
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/test_helper.h"
#include "lite/core/device_info.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(input_shape,
              "1,3,224,224",
              "input shapes, separated by colon and comma");

namespace paddle {
namespace lite_api {

void OutputOptModel(const std::string& load_model_dir,
                    const std::string& save_optimized_model_dir,
                    const std::vector<std::vector<int64_t>>& input_shapes) {
  lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
  config.set_preferred_place(Place{TARGET(kX86), PRECISION(kFloat)});
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
  auto predictor = lite_api::CreatePaddlePredictor(config);

  // delete old optimized model
  int ret = system(
      paddle::lite::string_format("rm -rf %s", save_optimized_model_dir.c_str())
          .c_str());
  if (ret == 0) {
    LOG(INFO) << "delete old optimized model " << save_optimized_model_dir;
  }
  predictor->SaveOptimizedModel(save_optimized_model_dir,
                                LiteModelType::kNaiveBuffer);
  LOG(INFO) << "Load model from " << load_model_dir;
  LOG(INFO) << "Save optimized model to " << save_optimized_model_dir;
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
void Run(const std::vector<std::vector<int64_t>>& input_shapes,
         const std::string& model_dir,
         const int repeat,
         const int thread_num,
         const int warmup_times = 0) {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Init();
  lite::DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, thread_num);
#endif
  lite_api::MobileConfig config;
  config.set_model_dir(model_dir);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  for (int j = 0; j < input_shapes.size(); ++j) {
    auto input_tensor = predictor->GetInput(j);
    input_tensor->Resize(input_shapes[j]);
    auto input_data = input_tensor->mutable_data<float>();
    int input_num = 1;
    for (int i = 0; i < input_shapes[j].size(); ++i) {
      input_num *= input_shapes[j][i];
    }
    for (int i = 0; i < input_num; ++i) {
      input_data[i] = 1.f;
    }
  }

  for (int i = 0; i < warmup_times; ++i) {
    predictor->Run();
  }

  auto start = lite::GetCurrentUS();
  for (int i = 0; i < repeat; ++i) {
    predictor->Run();
  }
  auto end = lite::GetCurrentUS();

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << thread_num
            << ", warmup: " << warmup_times << ", repeats: " << repeat
            << ", spend " << (end - start) / repeat / 1000.0
            << " ms in average.";

  auto output = predictor->GetOutput(0);
  auto out = output->data<float>();
  LOG(INFO) << "out " << out[0];
  LOG(INFO) << "out " << out[1];
  auto output_shape = output->shape();
  int output_num = 1;
  for (int i = 0; i < output_shape.size(); ++i) {
    output_num *= output_shape[i];
  }
  LOG(INFO) << "output_num: " << output_num;
}
#endif

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(INFO) << "usage: "
              << "--model_dir /path/to/your/model";
    exit(0);
  }
  std::string save_optimized_model_dir = FLAGS_model_dir + "opt2";

  auto split_string =
      [](const std::string& str_in) -> std::vector<std::string> {
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
  };

  auto get_shape = [](const std::string& str_shape) -> std::vector<int64_t> {
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
  };

  LOG(INFO) << "input shapes: " << FLAGS_input_shape;
  std::vector<std::string> str_input_shapes = split_string(FLAGS_input_shape);
  std::vector<std::vector<int64_t>> input_shapes;
  for (int i = 0; i < str_input_shapes.size(); ++i) {
    LOG(INFO) << "input shape: " << str_input_shapes[i];
    input_shapes.push_back(get_shape(str_input_shapes[i]));
  }

  // Output optimized model
  paddle::lite_api::OutputOptModel(
      FLAGS_model_dir, save_optimized_model_dir, input_shapes);

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  paddle::lite_api::Run(input_shapes,
                        save_optimized_model_dir,
                        FLAGS_repeats,
                        FLAGS_threads,
                        FLAGS_warmup);
#endif
  return 0;
}
