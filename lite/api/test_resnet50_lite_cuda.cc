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

#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/target_wrapper.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

void RunModel(const lite_api::CxxConfig& config) {
  auto predictor = lite_api::CreatePaddlePredictor(config);
  const int batch_size = 4;
  const int channels = 3;
  const int height = 224;
  const int width = 224;

  auto input_tensor = predictor->GetInput(0);
  std::vector<int64_t> input_shape{batch_size, channels, height, width};
  input_tensor->Resize(input_shape);
  std::vector<float> in_data(batch_size * channels * height * width);
  for (size_t i = 0; i < in_data.size(); i++) {
    in_data[i] = 1;
  }
  input_tensor->CopyFromCpu<float, lite_api::TargetType::kCUDA>(in_data.data());
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  std::vector<float> results{
      0.000241399, 0.000224183, 0.000536607, 0.000286386, 0.000726817,
      0.000212999, 0.00638716,  0.00128127,  0.000135354, 0.000767598,
      0.000241399, 0.000224183, 0.000536607, 0.000286386, 0.000726817,
      0.000212999, 0.00638716,  0.00128127,  0.000135354, 0.000767598,
      0.000241399, 0.000224183, 0.000536607, 0.000286386, 0.000726817,
      0.000212999, 0.00638716,  0.00128127,  0.000135354, 0.000767598,
      0.000241399, 0.000224183, 0.000536607, 0.000286386, 0.000726817,
      0.000212999, 0.00638716,  0.00128127,  0.000135354, 0.000767598};
  auto out = predictor->GetOutput(0);
  ASSERT_EQ(out->shape().size(), 2u);
  ASSERT_EQ(out->shape()[0], batch_size);
  ASSERT_EQ(out->shape()[1], 1000);
  std::vector<int64_t> shape = out->shape();
  int out_num =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<float> out_cpu(out_num);
  out->CopyToCpu(out_cpu.data());
  int step = 100;
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_NEAR(out_cpu[i * step], results[i], 1e-6);
  }
}

TEST(Resnet50, config_no_stream) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kCUDA), PRECISION(kFloat)}});

  RunModel(config);
}

TEST(Resnet50, config_exec_stream) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kCUDA), PRECISION(kFloat)}});
  cudaStream_t stream;
  lite::TargetWrapperCuda::CreateStream(&stream);
  config.set_cuda_stream(&stream);

  RunModel(config);
}

TEST(Resnet50, config_all_stream) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kCUDA), PRECISION(kFloat)}});
  cudaStream_t exec_stream;
  lite::TargetWrapperCuda::CreateStream(&exec_stream);
  cudaStream_t io_stream;
  lite::TargetWrapperCuda::CreateStream(&io_stream);
  config.set_cuda_stream(&exec_stream, &io_stream);

  RunModel(config);
}

TEST(Resnet50, config_multi_exec_stream) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kCUDA), PRECISION(kFloat)}});
  config.set_cuda_use_multi_stream(true);

  RunModel(config);
}

TEST(Resnet50, config_error) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kCUDA), PRECISION(kFloat)}});
  config.set_cuda_use_multi_stream(true);
  cudaStream_t exec_stream;
  lite::TargetWrapperCuda::CreateStream(&exec_stream);
  config.set_cuda_stream(&exec_stream);

  ASSERT_DEATH(RunModel(config), "");
}

}  // namespace lite
}  // namespace paddle
