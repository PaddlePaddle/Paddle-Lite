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

#ifndef LITE_API_TOOLS_BENCHMARK_BENCHMARK_H_
#define LITE_API_TOOLS_BENCHMARK_BENCHMARK_H_
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/tools/benchmark/utils/flags.h"
#include "lite/api/tools/opt_base.h"
#include "lite/core/device_info.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/model_util.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite_api {

class PerfData {
 public:
  void init(const int repeats) { repeats_ = repeats; }
  const float init_time() const { return init_time_; }
  const float first_time() const { return run_time_.at(0); }
  const float avg_pre_process_time() const {
    return std::accumulate(pre_process_time_.end() - repeats_,
                           pre_process_time_.end(),
                           0.f) /
           repeats_;
  }
  const float min_pre_process_time() const {
    return *std::min_element(pre_process_time_.end() - repeats_,
                             pre_process_time_.end());
  }
  const float max_pre_process_time() const {
    return *std::max_element(pre_process_time_.end() - repeats_,
                             pre_process_time_.end());
  }
  const float avg_post_process_time() const {
    return std::accumulate(post_process_time_.end() - repeats_,
                           post_process_time_.end(),
                           0.f) /
           repeats_;
  }
  const float min_post_process_time() const {
    return *std::min_element(post_process_time_.end() - repeats_,
                             post_process_time_.end());
  }
  const float max_post_process_time() const {
    return *std::max_element(post_process_time_.end() - repeats_,
                             post_process_time_.end());
  }
  const float avg_run_time() const {
    return std::accumulate(run_time_.end() - repeats_, run_time_.end(), 0.f) /
           repeats_;
  }
  const float min_run_time() const {
    return *std::min_element(run_time_.end() - repeats_, run_time_.end());
  }
  const float max_run_time() const {
    return *std::max_element(run_time_.end() - repeats_, run_time_.end());
  }

  void set_init_time(const float ms) { init_time_ = ms; }
  void set_pre_process_time(const float ms) { pre_process_time_.push_back(ms); }
  void set_post_process_time(const float ms) {
    post_process_time_.push_back(ms);
  }
  void set_run_time(const float ms) { run_time_.push_back(ms); }

 private:
  int repeats_{0};
  float init_time_{0.f};
  std::vector<float> pre_process_time_;
  std::vector<float> post_process_time_;
  std::vector<float> run_time_;
};

int Benchmark(int argc, char** argv);
void Run(const std::string& model_file,
         const std::vector<std::vector<int64_t>>& input_shape);

#ifdef __ANDROID__
std::string GetDeviceInfo() {
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

void StoreBenchmarkResult(const std::string res) {
  if (!FLAGS_result_path.empty()) {
    static bool first_call_flag = true;
    auto openmode = std::ios::ate;
    if (!first_call_flag) {
      openmode = std::ios::app;
    }
    std::ofstream fs(FLAGS_result_path, openmode);
    if (!fs.is_open()) {
      std::cerr << "Fail to open result file: " << FLAGS_result_path
                << std::endl;
    }
    fs << res;
    fs.close();
    first_call_flag = false;
  }
}

bool CheckFlagsValid() {
  bool ret = true;
  bool is_opt_model =
      (FLAGS_uncombined_model_dir.empty() && FLAGS_model_file.empty() &&
       FLAGS_param_file.empty() && !FLAGS_optimized_model_file.empty());
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
  if (!FLAGS_input_data_path.empty()) {
    auto paths = lite::Split(FLAGS_input_data_path, ":");
    auto shapes = lite::Split(FLAGS_input_shape, ":");
    if (paths.size() != shapes.size()) {
      std::cerr << lite::string_format(
                       "Option invalid: --input_data_path=%s  --input_shape=%s "
                       "\nThe num of input tensors is ambiguous.",
                       FLAGS_input_data_path.c_str(),
                       FLAGS_input_shape.c_str())
                << std::endl;
      ret = false;
    }
  }
  if (!FLAGS_validation_set.empty()) {
    if (FLAGS_config_path.empty()) {
      std::cerr
          << "Both --validation_set and --config_path options should be set!"
          << std::endl;
      ret = false;
    }
  }

  return ret;
}

const std::string PrintUsage() {
  std::stringstream ss;
  ss << "\nNote: \n"
        "  If load the optimized model, set --optimized_model_file. "
        "\n"
        "  Otherwise, set --uncombined_model_dir for uncomnbined paddle model "
        "or set --model_file and param_file for combined paddle model. \n"
        "For example (Android): \n"
        "  For optimized model : ./benchmark_bin "
        "--optimized_model_file=/data/local/tmp/mobilenetv1_opt.nb "
        "--input_shape=1,3,224,224 --backend=arm \n\n"
        "  For uncombined model: ./benchmark_bin "
        "--uncombined_model_dir=/data/local/tmp/mobilenetv1 "
        "--input_shape=1,3,224,224 --backend=arm \n\n"
        "  For combined model  : ./benchmark_bin "
        "--model_file=/data/local/tmp/mobilenetv1/model "
        "--param_file=/data/local/tmp/mobilenetv1/params "
        "--input_shape=1,3,224,224 --backend=arm \n\n"
        "For example (Linux/OSX): \n"
        "  You should add the directory of libmklml_intel.so/libmklml.dylib. "
        "to LD_LIBRARY_PATH before running benchmark_bin as following because "
        "benchmark_bin on Linux is dependent on it. \n"
        "  export "
        "LD_LIBRARY_PATH=./build.lite.linux*/third_party/install/mklml/lib/"
        ":$LD_LIBRARY_PATH \n"
        "  For optimized model : ./benchmark_bin "
        "--optimized_model_file=/path/to/mbilenetv1_opt.nb "
        "--input_shape=1,3,224,224 --backend=x86 \n\n"
        "  For uncombined model: ./benchmark_bin "
        "--uncombined_model_dir=/path/to/mobilenetv1 "
        "--input_shape=1,3,224,224 --backend=x86 \n\n"
        "  For combined model  : ./benchmark_bin "
        "--model_file=/path/to/mobilenetv1/model "
        "--param_file=/path/to/mobilenetv1/params "
        "--input_shape=1,3,224,224 --backend=x86 \n\n"
        "For detailed usage info: ./benchmark_bin --help \n\n";

  return ss.str();
}

void SetBackendConfig(lite_api::MobileConfig& config) {  // NOLINT
  if (FLAGS_backend == "opencl,arm" || FLAGS_backend == "opencl" ||
      FLAGS_backend == "opencl,x86" || FLAGS_backend == "x86_opencl") {
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
#ifdef __ANDROID__
    if (FLAGS_opencl_cache_dir.empty()) {
      FLAGS_opencl_cache_dir = "/data/local/tmp/";
    }
#endif  // __ANDROID__
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

  // nnadapter option
  std::vector<std::string> nnadapter_backends = {"imagination_nna",
                                                 "rockchip_npu",
                                                 "mediatek_apu",
                                                 "huawei_kirin_npu",
                                                 "huawei_ascend_npu",
                                                 "amlogic_npu",
                                                 "verisilicon_timvx"};
  auto backends_list = lite::Split(FLAGS_backend, ",");
  bool with_nnadapter =
      std::find(backends_list.begin(), backends_list.end(), "nnadapter") !=
      backends_list.end();
  if (with_nnadapter) {
    std::vector<std::string> nnadapter_devices;
    auto device_list = lite::Split(FLAGS_nnadapter_device_names, ",");

    if (device_list.size() < 1) {
      std::cerr << "The device list for nnadapter is null!" << std::endl;
      return;
    }

    for (auto& device : device_list) {
      if (std::find(nnadapter_backends.begin(),
                    nnadapter_backends.end(),
                    device) != nnadapter_backends.end()) {
        nnadapter_devices.push_back(device);
      } else {
        std::cerr << "Find and ignore unsupport nnadapter device: " << device
                  << std::endl;
      }
    }

    if (nnadapter_devices.size() == 0) {
      std::cerr << "No avaliable device found for nnadapter" << std::endl;
      return;
    } else {
      config.set_nnadapter_device_names(nnadapter_devices);
      config.set_nnadapter_context_properties(
          FLAGS_nnadapter_context_properties);
    }
  }
}

const std::string OutputOptModel(const std::string& opt_model_file) {
  auto opt = paddle::lite_api::OptBase();

  if (FLAGS_backend != "") {
    auto backends_list = lite::Split(FLAGS_backend, ",");
    bool with_nnadapter =
        std::find(backends_list.begin(), backends_list.end(), "nnadapter") !=
        backends_list.end();
    if (FLAGS_cpu_precision == "fp16") opt.EnableFloat16();
    if (with_nnadapter) {
      std::string valid_places;
      for (auto& backend : backends_list) {
        if (backend != "nnadapter") {
          valid_places += backend;
          valid_places += ',';
        }
      }
      if (FLAGS_nnadapter_device_names != "") {
        valid_places += FLAGS_nnadapter_device_names;
      } else {
        valid_places.pop_back();
      }
      opt.SetValidPlaces(valid_places);
    } else {
      opt.SetValidPlaces(FLAGS_backend);
    }
  }
  bool is_opt_model =
      (FLAGS_uncombined_model_dir.empty() && FLAGS_model_file.empty() &&
       FLAGS_param_file.empty() && !FLAGS_optimized_model_file.empty());
  if (is_opt_model) {
    if (!paddle::lite::IsFileExists(opt_model_file)) {
      std::cerr << lite::string_format("Mode file[%s] not exist!",
                                       opt_model_file.c_str())
                << std::endl;
      std::abort();
    }
    return FLAGS_optimized_model_file;
  }

  std::string model_dir = FLAGS_uncombined_model_dir;
  if (!FLAGS_uncombined_model_dir.empty()) {
    opt.SetModelDir(FLAGS_uncombined_model_dir);
  } else {
    model_dir = FLAGS_model_file.substr(0, FLAGS_model_file.rfind("/"));
    opt.SetModelFile(FLAGS_model_file);
    opt.SetParamFile(FLAGS_param_file);
  }
  auto npos = opt_model_file.find(".nb");
  std::string out_name = opt_model_file.substr(0, npos);
  if (out_name.empty()) {
    out_name = model_dir + "/opt";
  }
  opt.SetOptimizeOut(out_name);

  std::string saved_opt_model_file =
      opt_model_file.empty() ? out_name + ".nb" : opt_model_file;
  if (paddle::lite::IsFileExists(saved_opt_model_file)) {
    int err = system(
        lite::string_format("rm -rf %s", saved_opt_model_file.c_str()).c_str());
    if (err == 0) {
      std::cout << "Delete previous optimized model: " << saved_opt_model_file
                << std::endl;
    }
  }

  opt.Run();

  std::stringstream ss;
  ss << "\n======= Opt Info =======\n";
  ss << "Load paddle model from "
     << (FLAGS_uncombined_model_dir.empty()
             ? FLAGS_model_file + " and " + FLAGS_param_file
             : FLAGS_uncombined_model_dir)
     << std::endl;
  ss << "Save optimized model to " << saved_opt_model_file << std::endl;
  std::cout << ss.str();

  StoreBenchmarkResult(ss.str());
  return saved_opt_model_file;
}

}  // namespace lite_api
}  // namespace paddle

#endif  // LITE_API_TOOLS_BENCHMARK_BENCHMARK_H_
