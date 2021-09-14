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
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

TEST(Mobilenet_v1, test_mobilenetv1_lite_x86) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);

#ifdef LITE_WITH_OPENCL
  config.set_valid_places(
      {Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},
       Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
       Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},
       Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},
       Place{TARGET(kOpenCL), PRECISION(kInt32), DATALAYOUT(kNCHW)},
       Place{TARGET(kX86), PRECISION(kFloat)},
       Place{TARGET(kHost), PRECISION(kFloat)}});

  bool is_opencl_backend_valid =
      ::IsOpenCLBackendValid(false /*check_fp16_valid = false*/);
  std::cout << "is_opencl_backend_valid:" << is_opencl_backend_valid
            << std::endl;

  // Set opencl kernel binary.
  // Large addtitional prepare time is cost due to algorithm selecting and
  // building kernel from source code.
  // Prepare time can be reduced dramitically after building algorithm file
  // and OpenCL kernel binary on the first running.
  // The 1st running time will be a bit longer due to the compiling time if
  // you don't call `set_opencl binary_path_name` explicitly.
  // So call `set_opencl binary_path_name` explicitly is strongly recommended.

  // Make sure you have write permission of the binary path.
  // We strongly recommend each model has a unique binary name.
  const std::string bin_path = "./";
  const std::string bin_name = "lite_opencl_kernel.bin";
  config.set_opencl_binary_path_name(bin_path, bin_name);
  // CL_TUNE_NONE: 0
  // CL_TUNE_RAPID: 1
  // CL_TUNE_NORMAL: 2
  // CL_TUNE_EXHAUSTIVE: 3
  config.set_opencl_tune(CL_TUNE_NONE);
  // opencl precision option. Most x86 devices only support fp32, so set
  // CL_PRECISION_FP32 as default.
  // CL_PRECISION_AUTO: 0, first fp16 if valid, default
  // CL_PRECISION_FP32: 1, force fp32
  // CL_PRECISION_FP16: 2, force fp16
  config.set_opencl_precision(CL_PRECISION_FP32);
#else
  config.set_valid_places({lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
#endif

  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  std::vector<int64_t> input_shape{1, 3, 224, 224};
  input_tensor->Resize(input_shape);
  auto* data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  for (int i = 0; i < input_num; i++) {
    data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";
  std::vector<std::vector<float>> results;
  // i = 1
  results.emplace_back(std::vector<float>(
      {0.00019130898f, 9.467885e-05f,  0.00015971427f, 0.0003650665f,
       0.00026431272f, 0.00060884043f, 0.0002107942f,  0.0015819625f,
       0.0010323516f,  0.00010079765f, 0.00011006987f, 0.0017364529f,
       0.0048292773f,  0.0013995157f,  0.0018453331f,  0.0002428986f,
       0.00020211363f, 0.00013668182f, 0.0005855956f,  0.00025901722f}));
  auto out = predictor->GetOutput(0);
  ASSERT_EQ(out->shape().size(), 2u);
  ASSERT_EQ(out->shape()[0], 1);
  ASSERT_EQ(out->shape()[1], 1000);

#ifdef LITE_WITH_AVX
  const float abs_error = 1e-2;
#else
  const float abs_error = 1e-6;
#endif

  int step = 50;
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(out->data<float>()[j * step + (out->shape()[1] * i)],
                  results[i][j],
                  abs_error);
    }
  }
}

}  // namespace lite
}  // namespace paddle
