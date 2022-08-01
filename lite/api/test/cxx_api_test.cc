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

#include "lite/api/cxx_api.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

// For training.
DEFINE_string(startup_program_path, "", "");
DEFINE_string(main_program_path, "", "");
DEFINE_string(model_dir, "", "");
DEFINE_string(optimized_model, "", "");

namespace paddle {
namespace lite {

#ifndef LITE_WITH_ARM
TEST(CXXApi, test) {
  const lite::Tensor* out = RunHvyModel();
  LOG(INFO) << out << " memory size " << out->data_size();
  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << out->data<float>()[i];
  }
  LOG(INFO) << "dims " << out->dims();
  // LOG(INFO) << "out " << *out;
}

TEST(CXXApi, input_precision) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kX86), PRECISION(kFloat)}});
  predictor.Build(FLAGS_model_dir, "", "", valid_places);
  auto& precisions = predictor.GetInputPrecisions();
  ASSERT_EQ(precisions.size(), 1);
  ASSERT_EQ(precisions[0], PrecisionType::kFloat);
}

TEST(CXXApi, save_model) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kX86), PRECISION(kFloat)}});
  predictor.Build(FLAGS_model_dir, "", "", valid_places);

  LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
  predictor.SaveModel(FLAGS_optimized_model,
                      lite_api::LiteModelType::kProtobuf);
  predictor.SaveModel(FLAGS_optimized_model + ".naive",
                      lite_api::LiteModelType::kNaiveBuffer);
}

TEST(CXXApi, clone_predictor) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kX86), PRECISION(kFloat)}});
  predictor.Build(FLAGS_model_dir, "", "", valid_places);
  auto cloned_predictor = predictor.Clone();
  // primary predicotr
  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({1, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100; i++) {
    data[i] = 1;
  }

  predictor.Run();
  auto* output_tensor = predictor.GetOutput(0);
  auto output_shape = output_tensor->dims().Vectorize();
  ASSERT_EQ(output_shape.size(), 2);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 500);

  // cloned predictor
  auto* cloned_input_tensor = cloned_predictor->GetInput(0);
  cloned_input_tensor->Resize(std::vector<int64_t>({1, 100}));
  auto* cloned_data = cloned_input_tensor->mutable_data<float>();
  for (int i = 0; i < 100; i++) {
    cloned_data[i] = 1;
  }
  cloned_predictor->Run();
  auto* cloned_output_tensor = cloned_predictor->GetOutput(0);

  int step = 50;
  for (int i = 0; i < output_tensor->data_size(); i += step) {
    EXPECT_NEAR(output_tensor->data<float>()[i],
                cloned_output_tensor->data<float>()[i],
                1e-6);
  }
}

/*TEST(CXXTrainer, train) {
  Place place({TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW)});
  std::vector<Place> valid_places({place});
  auto scope = std::make_shared<lite::Scope>();

  CXXTrainer trainer(scope, valid_places);

  std::string main_program_pb, startup_program_pb;
  ReadBinaryFile(FLAGS_main_program_path, &main_program_pb);
  ReadBinaryFile(FLAGS_startup_program_path, &startup_program_pb);
  framework::proto::ProgramDesc main_program_desc, startup_program_desc;
  main_program_desc.ParseFromString(main_program_pb);
  startup_program_desc.ParseFromString(startup_program_pb);

  // LOG(INFO) << main_program_desc.DebugString();

  for (const auto& op : main_program_desc.blocks(0).ops()) {
    LOG(INFO) << "get op " << op.type();
  }

  return;

  trainer.RunStartupProgram(startup_program_desc);
  auto& exe = trainer.BuildMainProgramExecutor(main_program_desc);
  auto* tensor0 = exe.GetInput(0);
  tensor0->Resize(std::vector<int64_t>({100, 100}));
  auto* data0 = tensor0->mutable_data<float>();
  data0[0] = 0;

  exe.Run();
}*/
#endif

#ifdef LITE_WITH_ARM
TEST(CXXApi, save_model) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kARM), PRECISION(kFloat)}});
  predictor.Build(FLAGS_model_dir, "", "", valid_places);

  LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
  predictor.SaveModel(FLAGS_optimized_model,
                      lite_api::LiteModelType::kProtobuf);
  predictor.SaveModel(FLAGS_optimized_model + ".naive",
                      lite_api::LiteModelType::kNaiveBuffer);
}

TEST(CXXApi, load_model_naive) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kARM), PRECISION(kFloat)}});
  predictor.Build(FLAGS_optimized_model + ".naive.nb",
                  "",
                  "",
                  valid_places,
                  {},
                  lite_api::LiteModelType::kNaiveBuffer);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({1, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100; i++) {
    data[i] = 1;
  }

  predictor.Run();

  std::vector<float> result({0.4350058,
                             -0.6048313,
                             -0.29346266,
                             0.40377066,
                             -0.13400325,
                             0.37114543,
                             -0.3407839,
                             0.14574292,
                             0.4104212,
                             0.8938774});

  auto* output_tensor = predictor.GetOutput(0);
  auto output_shape = output_tensor->dims().Vectorize();
  ASSERT_EQ(output_shape.size(), 2);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 500);

  int step = 50;
  for (int i = 0; i < result.size(); i += step) {
    EXPECT_NEAR(output_tensor->data<float>()[i], result[i], 1e-6);
  }
}
#endif

}  // namespace lite
}  // namespace paddle
