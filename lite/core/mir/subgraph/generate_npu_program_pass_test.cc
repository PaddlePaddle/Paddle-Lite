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
#include <gtest/gtest.h>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/subgraph/subgraph_program_pass.h"
#include "lite/core/program.h"
#include "lite/core/tensor.h"

#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/core/op_registry.h"

#include "lite/model_parser/pb/program_desc.h"

DEFINE_string(model_dir, "", "model_dir");
DEFINE_string(optimized_model, "", "optimized_model");

namespace paddle {
namespace lite {

TEST(NPUSubgraph, mobilenetv1) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kARM), PRECISION(kFloat)},
                                   Place{TARGET(kNPU), PRECISION(kFloat)}});
  predictor.Build(
      FLAGS_model_dir, Place{TARGET(kARM), PRECISION(kFloat)}, valid_places);

  auto* input_tensor = predictor.GetInput(0);
  //   input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224,
  //   224})));
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 13, 1, 1})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  predictor.GenNPURuntimeProgram();

  predictor.Run();

  LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
  predictor.SaveModel(FLAGS_optimized_model);
}

}  // namespace lite
}  // namespace paddle
