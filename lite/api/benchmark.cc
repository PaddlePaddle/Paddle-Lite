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
#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/core/device_info.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(model_dir,
              "",
              "the path of the model, the model and param files is under "
              "model_dir.");
DEFINE_string(model_filename,
              "",
              "the filename of model file. When the model is combined formate, "
              "please set model_file.");
DEFINE_string(param_filename,
              "",
              "the filename of param file, set param_file when the model is "
              "combined formate.");
DEFINE_string(input_shape,
              "1,3,224,224",
              "set input shapes according to the model, "
              "separated by colon and comma, "
              "such as 1,3,244,244");
DEFINE_string(input_img_path,
              "",
              "the path of input image, if not set "
              "input_img_path, the input of model will be 1.0.");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_int32(power_mode,
             3,
             "arm power mode: "
             "0 for big cluster, "
             "1 for little cluster, "
             "2 for all cores, "
             "3 for no bind");
DEFINE_int32(threads, 1, "threads num");
DEFINE_string(result_filename,
              "result.txt",
              "save the inference time to the file.");
DEFINE_bool(run_model_optimize,
            false,
            "if set true, apply model_optimize_tool to "
            "model and use optimized model to test. ");

namespace paddle {
namespace lite_api {

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void OutputOptModel(const std::string& save_optimized_model_dir) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  if (!FLAGS_model_filename.empty() && !FLAGS_param_filename.empty()) {
    config.set_model_file(FLAGS_model_dir + "/" + FLAGS_model_filename);
    config.set_param_file(FLAGS_model_dir + "/" + FLAGS_param_filename);
  }
  std::vector<Place> vaild_places = {
      Place{TARGET(kARM), PRECISION(kFloat)},
  };
  config.set_valid_places(vaild_places);
  auto predictor = lite_api::CreatePaddlePredictor(config);

  int ret = system(
      paddle::lite::string_format("rm -rf %s", save_optimized_model_dir.c_str())
          .c_str());
  if (ret == 0) {
    LOG(INFO) << "Delete old optimized model " << save_optimized_model_dir;
  }
  predictor->SaveOptimizedModel(save_optimized_model_dir,
                                LiteModelType::kNaiveBuffer);
  LOG(INFO) << "Load model from " << FLAGS_model_dir;
  LOG(INFO) << "Save optimized model to " << save_optimized_model_dir;
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
void Run(const std::vector<int64_t>& input_shape,
         const std::string& model_dir,
         const std::string model_name) {
  // set config and create predictor
  lite_api::MobileConfig config;
  config.set_threads(FLAGS_threads);
  config.set_power_mode(static_cast<PowerMode>(FLAGS_power_mode));
  config.set_model_from_file(model_dir + ".nb");

  auto predictor = lite_api::CreatePaddlePredictor(config);

  // set input
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(input_shape);
  auto input_data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  if (FLAGS_input_img_path.empty()) {
    for (int i = 0; i < input_num; ++i) {
      input_data[i] = 1.f;
    }
  } else {
    std::fstream fs(FLAGS_input_img_path);
    if (!fs.is_open()) {
      LOG(FATAL) << "open input image " << FLAGS_input_img_path << " error.";
    }
    for (int i = 0; i < input_num; i++) {
      fs >> input_data[i];
    }
    // LOG(INFO) << "input data:" << input_data[0] << " " <<
    // input_data[input_num-1];
  }

  // warmup
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  // run
  std::vector<float> perf_vct;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start = GetCurrentUS();
    predictor->Run();
    auto end = GetCurrentUS();
    perf_vct.push_back((end - start) / 1000.0);
  }
  std::sort(perf_vct.begin(), perf_vct.end());
  float min_res = perf_vct.back();
  float max_res = perf_vct.front();
  float total_res = accumulate(perf_vct.begin(), perf_vct.end(), 0.0);
  float avg_res = total_res / FLAGS_repeats;

  // save result
  std::ofstream ofs(FLAGS_result_filename, std::ios::app);
  if (!ofs.is_open()) {
    LOG(FATAL) << "open result file failed";
  }
  ofs.precision(5);
  ofs << std::setw(30) << std::fixed << std::left << model_name;
  ofs << "min = " << std::setw(12) << min_res;
  ofs << "max = " << std::setw(12) << max_res;
  ofs << "average = " << std::setw(12) << avg_res;
  ofs << std::endl;
  ofs.close();
}
#endif

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(INFO) << "Please run ./benchmark_bin --help to obtain usage.";
    exit(0);
  }

  if (FLAGS_model_dir.back() == '/') {
    FLAGS_model_dir.pop_back();
  }
  std::size_t found = FLAGS_model_dir.find_last_of("/");
  std::string model_name = FLAGS_model_dir.substr(found + 1);
  std::string save_optimized_model_dir = FLAGS_model_dir + "_opt2";

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

  std::vector<int64_t> input_shape = get_shape(FLAGS_input_shape);

  // Output optimized model if needed
  if (FLAGS_run_model_optimize) {
    paddle::lite_api::OutputOptModel(save_optimized_model_dir);
  }

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  std::string run_model_dir =
      FLAGS_run_model_optimize ? save_optimized_model_dir : FLAGS_model_dir;
  paddle::lite_api::Run(input_shape, run_model_dir, model_name);
#endif
  return 0;
}
