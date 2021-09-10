// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_API_TOOLS_BENCHMARK_H_
#define LITE_API_TOOLS_BENCHMARK_H_
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/tools/flags.h"
#include "lite/api/tools/opt_base.h"
#include "lite/core/device_info.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite_api {

int Benchmark(int argc, char** argv);
void Run(const std::string& model_file,
         const std::vector<int64_t>& input_shape);

template <class T>
std::string Vector2Str(const std::vector<T>& input) {
  std::stringstream ss;
  for (int i = 0; i < input.size() - 1; i++) {
    ss << input[i] << ",";
  }
  ss << input.back();
  return ss.str();
}

template <class T>
T ShapeProduction(const std::vector<T>& shape) {
  T num = 1;
  for (auto i : shape) {
    num *= i;
  }
  return num;
}

std::string ShapePrint(const std::vector<shape_t>& shapes) {
  std::string shapes_str{""};
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    auto shape = shapes[shape_idx];
    std::string shape_str;
    for (auto i : shape) {
      shape_str += std::to_string(i) + ",";
    }
    shapes_str += shape_str;
    shapes_str +=
        (shape_idx != 0 && shape_idx == shapes.size() - 1) ? "" : " : ";
  }
  return shapes_str;
}

std::string ShapePrint(const shape_t& shape) {
  std::string shape_str{""};
  for (auto i : shape) {
    shape_str += std::to_string(i) + " ";
  }
  return shape_str;
}

std::vector<int64_t> GetInputShape(const std::string& str_shape) {
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
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += std::pow((in[i] - mean), 2);
  }
  variance /= length;
  return std::sqrt(variance);
}

#ifdef __ANDROID__
std::string get_device_info() {
  auto get_cmd_result = [](const std::string cmd) -> std::string {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
      std::cerr << "Could not get command return data!" << std::endl;
    }
    char ret[1024];
    fgets(ret, sizeof(ret), pipe);
    pclose(pipe);
    return std::string(ret);
  };

  std::string ret;
  std::string cmd;
  cmd = "getprop ro.product.vendor.brand";
  ret += "Brand: " + get_cmd_result(cmd);
  cmd = "getprop ro.product.vendor.device";
  ret += "Device: " + get_cmd_result(cmd);
  cmd = "getprop ro.product.vendor.model";
  ret += "Model: " + get_cmd_result(cmd);
  cmd = "getprop ro.build.version.release";
  ret += "Android Version: " + get_cmd_result(cmd);
  cmd = "getprop ro.build.version.sdk";
  ret += "Android API Level: " + get_cmd_result(cmd);

  return ret;
}
#endif  // __ANDROID__

bool CheckFlagsValid() {
  bool ret = true;
  bool is_opt_model =
      (FLAGS_uncombined_model_dir.empty() && FLAGS_model_file.empty() &&
       FLAGS_param_file.empty() && !FLAGS_optimized_model_path.empty());
  bool is_origin_model =
      (!FLAGS_uncombined_model_dir.empty() ||
       (!FLAGS_model_file.empty() && !FLAGS_param_file.empty()));
  if (!is_origin_model && !is_opt_model) {
    std::cerr << "\nNo model path is set!" << std::endl;
    ret = false;
  }
  if (!FLAGS_uncombined_model_dir.empty() &&
      (!FLAGS_model_file.empty() && !FLAGS_param_file.empty())) {
    std::cerr << "Both --uncombined_model_dir and --model_file --param_file "
                 "are set. Only need to set one model format!"
              << std::endl;
    ret = false;
  }
  if (FLAGS_backend.empty()) {
    std::cerr << "Must set --backend option!" << std::endl;
    ret = false;
  }
  if (FLAGS_input_shape.empty()) {
    std::cerr << "Must set --input_shape option!" << std::endl;
    ret = false;
  }

  return ret;
}

const std::string PrintUsage() {
  STL::stringstream ss;
  ss << "\nNote: \n"
        "  If load the optimized model, set --optimized_model_path. "
        "\n"
        "  Otherwise, set --uncombined_model_dir for uncomnbined paddle model "
        "or set --model_file and param_file for combined paddle model. \n"
        "For example: \n"
        "  For optimized model : ./benchmark_bin "
        "--optimized_model_path=/data/local/tmp/mobilenetv1_opt.nb "
        "--input_shape=1,3,224,224 --backend=arm \n\n"
        "  For uncombined model: ./benchmark_bin "
        "--uncombined_model_dir=/data/local/tmp/mobilenetv1 "
        "--input_shape=1,3,224,224 --backend=arm \n\n"
        "  For combined model  : ./benchmark_bin "
        "--model_file=/data/local/tmp/mobilenetv1/model "
        "--param_file=/data/local/tmp/mobilenetv1/params "
        "--input_shape=1,3,224,224 --backend=arm \n\n"
        "For detailed usage info: ./benchmark_bin --help \n\n";

  return ss.str();
}

void SetBackendConfig(lite_api::MobileConfig& config) {  // NOLINT
  if (FLAGS_backend == "opencl" || FLAGS_backend == "x86_opencl") {
    // Set opencl kernel binary.
    // Large addtitional prepare time is cost due to algorithm selecting and
    // building kernel from source code.
    // Prepare time can be reduced dramitically after building algorithm file
    // and OpenCL kernel binary on the first running.
    // The 1st running time will be a bit longer due to the compiling time if
    // you don't call `set_opencl_binary_path_name` explicitly.
    // So call `set_opencl_binary_path_name` explicitly is strongly
    // recommended.

    // Make sure you have write permission of the binary path.
    // We strongly recommend each model has a unique binary name.
    config.set_opencl_binary_path_name(FLAGS_opencl_cache_dir,
                                       FLAGS_opencl_kernel_cache_file);

    // opencl tune option
    auto tune_mode = CL_TUNE_NONE;
    if (FLAGS_opencl_tune_mode == "none") {
      tune_mode = CL_TUNE_NONE;
    } else if (FLAGS_opencl_tune_mode == "normal") {
      tune_mode = CL_TUNE_NORMAL;
    } else if (FLAGS_opencl_tune_mode == "rapid") {
      tune_mode = CL_TUNE_RAPID;
    } else if (FLAGS_opencl_tune_mode == "exhaustive") {
      tune_mode = CL_TUNE_EXHAUSTIVE;
    } else {
      std::cerr << "Illegal opencl tune mode: " << FLAGS_opencl_tune_mode
                << std::endl;
    }
    config.set_opencl_tune(
        tune_mode, FLAGS_opencl_cache_dir, FLAGS_opencl_tuned_file);

    // opencl precision option
    auto gpu_precision = CL_PRECISION_FP16;
    if (FLAGS_gpu_precision == "auto") {
      gpu_precision = CL_PRECISION_AUTO;
    } else if (FLAGS_gpu_precision == "fp16") {
      gpu_precision = CL_PRECISION_FP16;
    } else if (FLAGS_gpu_precision == "fp32") {
      gpu_precision = CL_PRECISION_FP32;
    }
    config.set_opencl_precision(gpu_precision);
  }
}

void OutputOptModel(const std::string& save_optimized_model_path) {
  auto opt = paddle::lite_api::OptBase();

  if (!FLAGS_uncombined_model_dir.empty()) {
    opt.SetModelDir(FLAGS_uncombined_model_dir);
  } else {
    opt.SetModelFile(FLAGS_model_file);
    opt.SetParamFile(FLAGS_param_file);
  }
  opt.SetOptimizeOut(save_optimized_model_path);

  if (FLAGS_backend != "") {
    if (FLAGS_cpu_precision == "fp16") opt.EnableFloat16();
    opt.SetValidPlaces(FLAGS_backend);
  }

  auto previous_opt_model = save_optimized_model_path + ".nb";
  int ret = system(
      paddle::lite::string_format("rm -rf %s", previous_opt_model.c_str())
          .c_str());
  if (ret == 0) {
    std::cout << "Delete previous optimized model: " << previous_opt_model
              << std::endl;
  }

  opt.Run();

  STL::stringstream ss;
  ss << "\n======= Opt Info =======\n";
  ss << "Load paddle model from "
     << (FLAGS_uncombined_model_dir.empty()
             ? FLAGS_model_file + " and " + FLAGS_param_file
             : FLAGS_uncombined_model_dir)
     << std::endl;
  ss << "Save optimized model to " << save_optimized_model_path + ".nb"
     << std::endl;
  std::cout << ss.str() << std::endl;
  std::ofstream ofs(FLAGS_result_path, std::ios::app);
  if (!ofs.is_open()) {
    std::cerr << "Fail to open result file: " << FLAGS_result_path << std::endl;
  }
  ofs << ss.str();
  ofs.close();
}

}  // namespace lite_api
}  // namespace paddle

#endif  // LITE_API_TOOLS_BENCHMARK_H_
