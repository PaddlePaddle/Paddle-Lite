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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/tests/utils/timer.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(input_shape,
              "1,3,224,224",
              "input shapes, separated by colon and comma");

DEFINE_int32(cluster,
             3,
             "arm power mode: 0 for big cluster, 1 for little cluster, 2 for "
             "all cores, 3 for no bind");

void paddle_infer(const std::string& model_dir,
                  const std::vector<std::vector<int64_t>>& input_shapes) {
  paddle::lite_api::MobileConfig cfg;
  cfg.set_model_dir(model_dir);
  cfg.set_power_mode(static_cast<paddle::lite_api::PowerMode>(FLAGS_cluster));
  cfg.set_threads(FLAGS_threads);
  LOG(INFO) << "set mobile config, model_dir:" << model_dir
            << ", cluster: " << FLAGS_cluster << ", threads: " << cfg.threads();
  auto predictor =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          cfg);
  for (int i = 0; i < input_shapes.size(); i++) {
    auto in = predictor->GetInput(i);
    in->Resize(input_shapes[i]);
    in->mutable_data<float>();
  }
  LOG(INFO) << "set input shape: ";

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }
  LOG(INFO) << "warmup";

  lite::test::Timer ti;
  for (int j = 0; j < FLAGS_repeats; ++j) {
    ti.start();
    predictor->Run();
    ti.end();
  }

  LOG(INFO) << "model in : " << model_dir;
  for (int k = 0; k < input_shapes.size(); ++k) {
    LOG(INFO) << "shape " << k << ": ";
    for (int i = 0; i < input_shapes[k].size(); ++i) {
      LOG(INFO) << input_shapes[k][i];
    }
  }
  LOG(INFO) << "avg time: " << ti.get_average_ms()
            << ", max time: " << ti.get_max_time()
            << ", min time: " << ti.get_min_time();
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(INFO) << "usage: "
              << "--model_dir /path/to/your/model";
    exit(0);
  }
  std::string model_dir = FLAGS_model_dir;

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

  paddle_infer(model_dir, input_shapes);

  return 0;
}
