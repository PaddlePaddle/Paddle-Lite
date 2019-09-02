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
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/core/tensor.h"

#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"

#include "lite/model_parser/pb/program_desc.h"

DEFINE_string(optimized_model, "", "optimized_model");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(im_channel, 3, "im_channel");

namespace paddle {
namespace lite {

void TestModel(lite::Predictor* predictor,
               const std::vector<Place>& valid_places,
               const std::string& model_dir) {
  predictor->Build(model_dir,
                   model_dir + "/model",
                   model_dir + "/params",
                   Place{TARGET(kARM), PRECISION(kFloat)},
                   valid_places);

  auto* input_tensor = predictor->GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>(
      {FLAGS_batch_size, FLAGS_im_channel, FLAGS_im_height, FLAGS_im_width})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  predictor->Run();
  if (model_dir != FLAGS_optimized_model &&
      std::find(valid_places.begin(),
                valid_places.end(),
                Place{TARGET(kNPU), PRECISION(kFloat)}) != valid_places.end()) {
    predictor->SaveModel(FLAGS_optimized_model);
  }
}

void CompareOutData(const lite::Predictor& tgt, const lite::Predictor& ref) {
  auto* tgt_otensor = tgt.GetOutput(0);
  auto* ref_otensor = ref.GetOutput(0);
  const auto* tgt_pdata = tgt_otensor->data<float>();
  const auto* ref_pdata = ref_otensor->data<float>();
  EXPECT_EQ(tgt_otensor->dims().production(), ref_otensor->dims().production());
  for (size_t i = 0; i < tgt_otensor->dims().production(); ++i) {
    auto diff = std::fabs((tgt_pdata[i] - ref_pdata[i]) / ref_pdata[i]);
    VLOG(3) << diff;
    EXPECT_LT(diff, 0.1);
  }
}

TEST(NPUSubgraph, compare) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, 1);

  lite::Predictor predictor_arm, predictor_npu, predictor_npu_savedmodel;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kARM), PRECISION(kFloat)}});

  TestModel(&predictor_arm, valid_places, FLAGS_model_dir);

  valid_places.push_back(Place{TARGET(kNPU), PRECISION(kFloat)});
  TestModel(&predictor_npu, valid_places, FLAGS_model_dir);

  CompareOutData(predictor_npu, predictor_arm);
  LOG(INFO) << " ================ NPU speed ================== ";
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start = GetCurrentUS();
    predictor_npu.Run();
    LOG(INFO) << i << ", " << GetCurrentUS() - start << "us";
  }

  LOG(INFO) << " =================== ARM CPU speed =================== ";
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start = GetCurrentUS();
    predictor_arm.Run();
    LOG(INFO) << i << ", " << GetCurrentUS() - start << "us";
  }

  TestModel(&predictor_npu_savedmodel, valid_places, FLAGS_optimized_model);

  CompareOutData(predictor_npu_savedmodel, predictor_arm);
}

}  // namespace lite
}  // namespace paddle
