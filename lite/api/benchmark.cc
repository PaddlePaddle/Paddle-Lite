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
#if !defined(_WIN32)
#include <sys/time.h>
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
#include "lite/backends/x86/port.h"
#endif
#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include <time.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/device_info.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(optimized_model_path,
              "",
              "the path of the model that is optimized by opt.");
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
DEFINE_bool(show_output, false, "Wether to show the output in shell.");

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
      Place{TARGET(kARM), PRECISION(kInt32)},
      Place{TARGET(kARM), PRECISION(kInt64)},
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

int64_t ShapeProduction(const std::vector<int64_t>& shape) {
  int64_t num = 1;
  for (auto i : shape) {
    num *= i;
  }
  return num;
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
void Run(const std::vector<int64_t>& input_shape,
         const std::string& model_path,
         const std::string model_name) {
  // set config and create predictor
  lite_api::MobileConfig config;
  config.set_threads(FLAGS_threads);
  config.set_power_mode(static_cast<PowerMode>(FLAGS_power_mode));
  config.set_model_from_file(model_path);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  // set input
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(input_shape);
  auto input_data = input_tensor->mutable_data<float>();
  int64_t input_num = ShapeProduction(input_shape);
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
  std::stable_sort(perf_vct.begin(), perf_vct.end());
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

  if (FLAGS_show_output) {
    auto out_tensor = predictor->GetOutput(0);
    auto* out_data = out_tensor->data<float>();
    int64_t output_num = ShapeProduction(out_tensor->shape());
    float max_value = out_data[0];
    int max_index = 0;
    for (int i = 0; i < output_num; i++) {
      if (max_value < out_data[i]) {
        max_value = out_data[i];
        max_index = i;
      }
    }
    LOG(INFO) << "max_value:" << max_value;
    LOG(INFO) << "max_index:" << max_index;
    LOG(INFO) << "output data[0:10]:";
    for (int i = 0; i < 10; i++) {
      LOG(INFO) << out_data[i];
    }
  }
}
#endif

}  // namespace lite_api
}  // namespace paddle

void print_usage() {
  std::string help_info =
      "Usage: \n"
      "./benchmark_bin \n"
      "  --optimized_model_path (The path of the model that is optimized\n"
      "    by opt. If the model is optimized, please set the param.) \n"
      "    type: string \n"
      "  --model_dir (The path of the model that is not optimized by opt,\n"
      "    the model and param files is under model_dir.) type: string \n"
      "  --model_filename (The filename of model file. When the model is\n "
      "    combined formate, please set model_file. Otherwise, it is not\n"
      "    necessary to set it.) type: string \n"
      "  --param_filename (The filename of param file, set param_file when\n"
      "    the model is combined formate. Otherwise, it is not necessary\n"
      "    to set it.) type: string \n"
      "  --input_shape (Set input shapes according to the model, separated by\n"
      "    colon and comma, such as 1,3,244,244) type: string\n"
      "    default: 1,3,224,224 \n"
      "  --input_img_path (The path of input image, if not set\n"
      "    input_img_path, the input will be 1.0.) type: string \n "
      "  --power_mode (Arm power mode: 0 for big cluster, 1 for little\n"
      "    cluster, 2 for all cores, 3 for no bind) type: int32 default: 3\n"
      "  --repeats (Repeats times) type: int32 default: 1 \n"
      "  --result_filename (Save the inference time to the file.) type: \n"
      "    string default: result.txt \n"
      "  --threads (Threads num) type: int32 default: 1 \n"
      "  --warmup (Warmup times) type: int32 default: 0 \n"
      "Note that: \n"
      "  If load the optimized model, set optimized_model_path. Otherwise, \n"
      "    set model_dir, model_filename and param_filename according to \n"
      "    the model. \n";
  LOG(INFO) << help_info;
}

int main(int argc, char** argv) {
  // Check inputs
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  bool is_opt_model = (FLAGS_optimized_model_path != "");
  bool is_origin_model = (FLAGS_model_dir != "");
  if (!is_origin_model && !is_opt_model) {
    LOG(INFO) << "Input error, the model path should not be empty.\n";
    print_usage();
    exit(0);
  }

  // Get input shape
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

  // Get model_name and run_model_path
  std::string model_name;
  std::string run_model_path;
  if (is_origin_model) {
    if (FLAGS_model_dir.back() == '/') {
      FLAGS_model_dir.pop_back();
    }
    std::size_t found = FLAGS_model_dir.find_last_of("/");
    model_name = FLAGS_model_dir.substr(found + 1);
    std::string optimized_model_path = FLAGS_model_dir + "_opt2";
    paddle::lite_api::OutputOptModel(optimized_model_path);
    run_model_path = optimized_model_path + ".nb";
  } else {
    size_t found1 = FLAGS_optimized_model_path.find_last_of("/");
    size_t found2 = FLAGS_optimized_model_path.find_last_of(".");
    size_t len = found2 - found1 - 1;
    model_name = FLAGS_optimized_model_path.substr(found1 + 1, len);
    run_model_path = FLAGS_optimized_model_path;
  }

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run test
  paddle::lite_api::Run(input_shape, run_model_path, model_name);
#endif
  return 0;
}
