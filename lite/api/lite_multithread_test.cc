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
#include "lite/core/profile/timer.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/basic_profiler.h"
#endif             // LITE_WITH_PROFILE
#include <thread>  // NOLINT

using paddle::lite::profile::Timer;

DEFINE_string(input_shape,
              "1,3,224,224",
              "input shapes, separated by colon and comma");

DEFINE_string(model_dir_0, "", "model_dir_0");
DEFINE_string(input_shape_0,
              "1,3,224,224",
              "input shapes another, separated by colon and comma");
DEFINE_string(target, "arm", "main target for Predictor: arm, opencl");
DEFINE_bool(use_optimize_nb,
            false,
            "optimized & naive buffer model for mobile devices");

DEFINE_int32(test_type, 0, "multithread test type");

namespace paddle {
namespace lite_api {

void OutputOptModel(const std::string& load_model_dir,
                    const std::string& save_optimized_model_dir,
                    const std::vector<std::vector<int64_t>>& input_shapes) {
  lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
  if (FLAGS_target == "arm") {
    config.set_valid_places({
        Place{TARGET(kARM), PRECISION(kFloat)},
    });
  } else if (FLAGS_target == "opencl") {
    config.set_valid_places({
        Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},
        Place{TARGET(kARM)},  // enable kARM CPU kernel when no opencl kernel
    });
  }
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
         const PowerMode power_mode,
         const int thread_num,
         const int repeat,
         int tid,
         const int warmup_times = 5) {
  lite_api::MobileConfig config;
  config.set_model_from_file(model_dir + ".nb");
  config.set_power_mode(power_mode);
  config.set_threads(thread_num);

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

  Timer ti;
  for (int j = 0; j < repeat; ++j) {
    ti.Start();
    predictor->Run();
    float t = ti.Stop();
    auto output = predictor->GetOutput(0);
    auto out = output->data<float>();
    LOG(INFO) << "[thread " << tid << "] Model: " << model_dir
              << " output[0]:" << out[0] << "; output[1]:" << out[1];
  }
  LOG(INFO) << "[thread " << tid << "] Model: " << model_dir
            << ", power_mode: " << static_cast<int>(power_mode)
            << ", threads num " << thread_num
            << ", avg time: " << ti.LapTimes().Avg() << "ms"
            << ", min time: " << ti.LapTimes().Min() << " ms"
            << ", max time: " << ti.LapTimes().Max() << " ms.";
}

void RunTestType_00(const std::vector<std::vector<int64_t>>& input_shapes,
                    const std::string& model_dir,
                    const PowerMode power_mode,
                    const int thread_num,
                    const int repeat,
                    const int warmup_times = 5) {
  std::thread run_th0(Run,
                      input_shapes,
                      model_dir,
                      power_mode,
                      thread_num,
                      repeat,
                      0,
                      warmup_times);
  Run(input_shapes, model_dir, power_mode, thread_num, repeat, 1, warmup_times);
  run_th0.join();
}
void RunTestType_01(const std::vector<std::vector<int64_t>>& input_shapes,
                    const std::string& model_dir,
                    const std::vector<std::vector<int64_t>>& input_shapes_0,
                    const std::string& model_dir_0,
                    const PowerMode power_mode,
                    const int thread_num,
                    const int repeat,
                    const int warmup_times = 5) {
  std::thread run_th0(Run,
                      input_shapes,
                      model_dir,
                      power_mode,
                      thread_num,
                      repeat,
                      0,
                      warmup_times);
  Run(input_shapes_0,
      model_dir_0,
      power_mode,
      thread_num,
      repeat,
      1,
      warmup_times);
  run_th0.join();
}

void run_with_predictor(std::shared_ptr<PaddlePredictor> predictor,
                        const std::vector<std::vector<int64_t>>& input_shapes,
                        int index,
                        const std::string& name) {
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

  Timer ti;
  ti.Start();
  predictor->Run();
  float t = ti.Stop();

  auto output = predictor->GetOutput(0);
  auto out = output->data<float>();
  LOG(INFO) << "[thread " << index << "] name: " << name
            << ",run time: " << ti.LapTimes().Avg() << "ms"
            << " output[0]:" << out[0] << "; output[1]:" << out[1];
}
void RunTestType_10(const std::vector<std::vector<int64_t>>& input_shapes,
                    const std::string& model_dir,
                    const PowerMode power_mode,
                    const int thread_num,
                    const int repeat,
                    int warmup = 5) {
  lite_api::MobileConfig config;
  config.set_model_from_file(model_dir + ".nb");
  config.set_power_mode(power_mode);
  config.set_threads(thread_num);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  for (int i = 0; i < repeat; ++i) {
    std::thread pre_th0(
        run_with_predictor, predictor, input_shapes, i, model_dir);
    pre_th0.join();
  }
}
void RunTestType_11(const std::vector<std::vector<int64_t>>& input_shapes,
                    const std::string& model_dir,
                    const std::vector<std::vector<int64_t>>& input_shapes_0,
                    const std::string& model_dir_0,
                    const PowerMode power_mode,
                    const int thread_num,
                    const int repeat,
                    int warmup = 5) {
  lite_api::MobileConfig config;
  config.set_model_from_file(model_dir + ".nb");
  config.set_power_mode(power_mode);
  config.set_threads(thread_num);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  config.set_model_from_file(model_dir_0 + ".nb");
  auto predictor_0 = lite_api::CreatePaddlePredictor(config);

  for (int i = 0; i < 2 * repeat; i += 2) {
    std::thread pre_th0(
        run_with_predictor, predictor, input_shapes, i, model_dir);
    std::thread pre_th1(
        run_with_predictor, predictor_0, input_shapes_0, i + 1, model_dir_0);
    pre_th0.join();
    pre_th1.join();
  }
}

#endif

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(INFO) << "usage: "
              << "--model_dir /path/to/your/model --model_dir_0 "
                 "/path/to/your/model0  --target `arm` or `opencl`";
    exit(0);
  }
  std::string save_optimized_model_dir = "";
  std::string save_optimized_model_dir_0 = "";
  if (FLAGS_use_optimize_nb) {
    save_optimized_model_dir = FLAGS_model_dir;
    save_optimized_model_dir_0 = FLAGS_model_dir_0;
  } else {
    save_optimized_model_dir = FLAGS_model_dir + "opt2";
    save_optimized_model_dir_0 = FLAGS_model_dir_0 + "opt2";
  }

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

  std::vector<std::string> str_input_shapes = split_string(FLAGS_input_shape);
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t i = 0; i < str_input_shapes.size(); ++i) {
    input_shapes.push_back(get_shape(str_input_shapes[i]));
  }
  std::vector<std::string> str_input_shapes_0 =
      split_string(FLAGS_input_shape_0);
  std::vector<std::vector<int64_t>> input_shapes_0;
  for (size_t i = 0; i < str_input_shapes_0.size(); ++i) {
    input_shapes_0.push_back(get_shape(str_input_shapes_0[i]));
  }

  if (!FLAGS_use_optimize_nb) {
    // Output optimized model
    paddle::lite_api::OutputOptModel(
        FLAGS_model_dir, save_optimized_model_dir, input_shapes);
    paddle::lite_api::OutputOptModel(
        FLAGS_model_dir_0, save_optimized_model_dir_0, input_shapes_0);
  }

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  if (FLAGS_test_type == 0) {
    paddle::lite_api::RunTestType_00(
        input_shapes,
        save_optimized_model_dir,
        static_cast<paddle::lite_api::PowerMode>(0),
        FLAGS_threads,
        FLAGS_repeats,
        5);
    LOG(INFO) << "=========above is case 0, below is case "
                 "1============================";
    paddle::lite_api::RunTestType_10(
        input_shapes,
        save_optimized_model_dir,
        static_cast<paddle::lite_api::PowerMode>(0),
        FLAGS_threads,
        FLAGS_repeats);
  }
  if (FLAGS_test_type == 1) {
    paddle::lite_api::RunTestType_01(
        input_shapes,
        save_optimized_model_dir,
        input_shapes_0,
        save_optimized_model_dir_0,
        static_cast<paddle::lite_api::PowerMode>(0),
        FLAGS_threads,
        FLAGS_repeats,
        5);
    LOG(INFO) << "=========above is case 0, below is case "
                 "1============================";
    paddle::lite_api::RunTestType_11(
        input_shapes,
        save_optimized_model_dir,
        input_shapes_0,
        save_optimized_model_dir_0,
        static_cast<paddle::lite_api::PowerMode>(0),
        FLAGS_threads,
        FLAGS_repeats);
  }

#endif
  return 0;
}
