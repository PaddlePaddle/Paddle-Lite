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
#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#endif
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(model_dir,
              "",
              "path of the model. This option will be ignored if model_file "
              "and param_file are exist");
DEFINE_string(model_file, "", "model file path of the combined-param model");
DEFINE_string(param_file, "", "param file path of the combined-param model");
DEFINE_string(
    optimize_out_type,
    "protobuf",
    "store type of the output optimized model. protobuf/naive_buffer");
DEFINE_string(optimize_out, "", "path of the output optimized model");
DEFINE_string(valid_targets,
              "arm",
              "The targets this model optimized for, should be one of (arm, "
              "opencl, x86), splitted by space");
DEFINE_bool(prefer_int8_kernel, false, "Prefer to run model with int8 kernels");

namespace paddle {
namespace lite_api {

void Main() {
  if (!FLAGS_model_file.empty() && !FLAGS_param_file.empty()) {
    LOG(WARNING)
        << "Load combined-param model. Option model_dir will be ignored";
  }

  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_model_file(FLAGS_model_file);
  config.set_param_file(FLAGS_param_file);

  std::vector<Place> valid_places;
  auto target_reprs = lite::Split(FLAGS_valid_targets, " ");
  for (auto& target_repr : target_reprs) {
    if (target_repr == "arm") {
      valid_places.emplace_back(TARGET(kARM));
    } else if (target_repr == "opencl") {
      valid_places.emplace_back(TARGET(kOpenCL));
    } else if (target_repr == "x86") {
      valid_places.emplace_back(TARGET(kX86));
    } else {
      LOG(FATAL) << lite::string_format(
          "Wrong target '%s' found, please check the command flag "
          "'valid_targets'",
          target_repr.c_str());
    }
  }

  CHECK(!valid_places.empty())
      << "At least one target should be set, should set the "
         "command argument 'valid_targets'";
  if (FLAGS_prefer_int8_kernel) {
    LOG(WARNING) << "Int8 mode is only support by ARM target";
    valid_places.push_back(Place{TARGET(kARM), PRECISION(kInt8)});
    config.set_preferred_place(Place{TARGET(kARM), PRECISION(kInt8)});
  }
  config.set_valid_places(valid_places);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  LiteModelType model_type;
  if (FLAGS_optimize_out_type == "protobuf") {
    model_type = LiteModelType::kProtobuf;
  } else if (FLAGS_optimize_out_type == "naive_buffer") {
    model_type = LiteModelType::kNaiveBuffer;
  } else {
    LOG(FATAL) << "Unsupported Model type :" << FLAGS_optimize_out_type;
  }

  predictor->SaveOptimizedModel(FLAGS_optimize_out, model_type);
}

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  paddle::lite_api::Main();
  return 0;
}
