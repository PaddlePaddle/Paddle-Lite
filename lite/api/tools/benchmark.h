// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
      LOG(ERROR) << "Could not get command return data.";
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

const std::string PrintUsage() {
  STL::stringstream ss;
  ss << "\nNote: \n"
        "  If load the optimized model, set --optimized_model_path. "
        "\n"
        "  Otherwise, set --uncombined_model_dir for uncomnbined paddle model "
        "or set --model_file and param_file for combined paddle model. \n"
        "For example: \n"
        "  For optimized model : ./benchmark_bin "
        "--optimized_model_path=/data/local/tmp/mobilenetv1_opt.nb \n"
        "  For uncombined model: ./benchmark_bin "
        "--uncombined_model_dir=/data/local/tmp/mobilenetv1 \n"
        "  For combined model  : ./benchmark_bin "
        "--model_file=/data/local/tmp/mobilenetv1/model "
        "--param_file=/data/local/tmp/mobilenetv1/params \n";

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
      LOG(ERROR) << "Illegal opencl tune mode: " << FLAGS_opencl_tune_mode;
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
  lite_api::CxxConfig config;
  if (!FLAGS_uncombined_model_dir.empty()) {
    config.set_model_dir(FLAGS_uncombined_model_dir);
  } else {
    config.set_model_file(FLAGS_model_file);
    config.set_param_file(FLAGS_param_file);
  }

  std::vector<Place> valid_places;
  if (FLAGS_backend == "opencl") {
    valid_places = {
        Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},
        Place{TARGET(kOpenCL), PRECISION(kInt32), DATALAYOUT(kNCHW)},
        Place{TARGET(kARM)},
    };
  } else if (FLAGS_backend == "x86_opencl") {
    valid_places = {
        Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},
        Place{TARGET(kOpenCL), PRECISION(kInt32), DATALAYOUT(kNCHW)},
        Place{TARGET(kX86), PRECISION(kFloat)},
        Place{TARGET(kHost), PRECISION(kFloat)},
    };
  } else {
    valid_places = {
        Place{TARGET(kARM), PRECISION(kInt32)},
        Place{TARGET(kARM), PRECISION(kInt64)},
    };
    if (FLAGS_cpu_precision == "fp16") {
      valid_places.emplace_back(Place{TARGET(kARM), PRECISION(kFP16)});
    }
    valid_places.emplace_back(Place{TARGET(kARM), PRECISION(kFloat)});
  }
  config.set_valid_places(valid_places);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  int ret = system(paddle::lite::string_format(
                       "rm -rf %s", save_optimized_model_path.c_str())
                       .c_str());
  if (ret == 0) {
    LOG(INFO) << "Delete old optimized model " << save_optimized_model_path;
  }
  predictor->SaveOptimizedModel(save_optimized_model_path,
                                LiteModelType::kNaiveBuffer);

  STL::stringstream ss;
  ss << "\n======= Opt Info =======\n";
  ss << "Load paddle model from "
     << (FLAGS_uncombined_model_dir.empty()
             ? FLAGS_model_file + " and " + FLAGS_param_file
             : FLAGS_uncombined_model_dir)
     << std::endl;
  ss << "Save optimized model to " << save_optimized_model_path + ".nb"
     << std::endl;
  LOG(INFO) << ss.str();
  std::ofstream ofs(FLAGS_result_path, std::ios::app);
  if (!ofs.is_open()) {
    LOG(FATAL) << "Fail to open result file: " << FLAGS_result_path;
  }
  ofs << ss.str();
  ofs.close();
}

}  // namespace lite_api
}  // namespace paddle

#endif  // LITE_API_TOOLS_BENCHMARK_H_
