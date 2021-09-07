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
#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
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
#include "lite/utils/timer.h"

// Model options
DEFINE_string(optimized_model_path,
              "",
              "The path of the model that is optimized by opt.");
DEFINE_string(uncombined_model_dir,
              "",
              "The dir of the uncombined model, the model and param files "
              "are under model_dir.");
DEFINE_string(model_file,
              "",
              "The filename of model file. Set model_file when the model is "
              "combined formate.");
DEFINE_string(param_file,
              "",
              "The filename of param file. Set param_file when the model is "
              "combined formate.");
DEFINE_string(input_shape,
              "1,3,224,224",
              "Set input shapes according to the model, "
              "separated by comma and colon, "
              "such as 1,3,244,244 for only one input, "
              "1,3,224,224:1,5 for two inputs.");
DEFINE_string(input_data_path,
              "",
              "The path of input image, if not set "
              "input_data_path, the input of model will be 1.0.");
DEFINE_bool(show_output_elem, false, "Show each output tensor's all elements.");

// Common runtime options
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_double(run_delay,
              -1.0,
              "The delay in seconds between subsequent benchmark runs. "
              "Non-positive values mean use no delay.");
DEFINE_int32(power_mode,
             0,
             "arm power mode: "
             "0 for big cluster, "
             "1 for little cluster, "
             "2 for all cores, "
             "3 for no bind");
DEFINE_int32(threads, 1, "threads num");
DEFINE_string(result_path, "result.txt", "Save benchmark info to the file.");

// Backend options
DEFINE_string(backend,
              "cpu",
              "To use a particular backend for execution, "
              "and fail if unsuccessful. Should be one of: cpu, opencl, metal, "
              "x86_opencl.");
DEFINE_string(cpu_precision,
              "fp32",
              "Register fp32 or fp16 arm-cpu kernel when optimized model. "
              "Should be one of: fp32, fp16.");
DEFINE_string(gpu_precision,
              "fp16",
              "Allow to process computation in lower precision in GPU. "
              "Should be one of: fp32, fp16.");
DEFINE_string(
    opencl_cache_dir,
    "",
    "A directory in which kernel binary and tuned file will be stored.");
DEFINE_string(opencl_kernel_cache_file,
              "paddle_lite_opencl_kernel.bin",
              "Set opencl kernel binary filename. "
              "We strongly recommend each model has a unique binary name.");
DEFINE_string(opencl_tuned_file,
              "paddle_lite_opencl_tuned.params",
              "Set opencl tuned filename."
              "We strongly recommend each model has a unique param name.");
DEFINE_string(opencl_tune_mode,
              "normal",
              "Set opencl tune option: none, rapid, normal, exhaustive.");

// Profiling options
DEFINE_bool(enable_op_time_profile,
            false,
            "Whether to run with op time profiling.");
DEFINE_bool(enable_memory_profile,
            false,
            "Whether to report the peak memory footprint by periodically "
            "checking the memory footprint. Internally, a separate thread "
            " will be spawned for this periodic check. Therefore, "
            "the performance benchmark result could be affected.");
DEFINE_int32(memory_footprint_check_interval_ms,
             5,
             "The interval in millisecond between two consecutive memory "
             "footprint checks. This is only used when "
             "--enable_memory_profile is set to true.");

// Others

namespace paddle {
namespace lite_api {

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
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
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
#endif

void PrintUsage() {
  std::string help_info =
      "Usage: \n"
      "./benchmark_bin \n"
      "  --optimized_model_path (The path of the model that is optimized\n"
      "    by opt. If the model is optimized, please set the param.) \n"
      "    type: string \n"
      "  --model_dir (The path of the model that is not optimized by opt,\n"
      "    the model and param files is under model_dir.) type: string \n"
      "  --model_filename (The filename of model file. When the model is \n"
      "    combined formate, please set model_file, such as `xx.pdmodel`. \n"
      "    Otherwise, it is not necessary to set it.) type: string \n"
      "  --params_filename (The filename of param file, set param_file when\n"
      "    the model is combined formate, such as `xx.pdiparams`. Otherwise, \n"
      "    it is not necessary to set it.) type: string \n"
      "  --input_shape (Set input shapes according to the model, separated by\n"
      "    colon and comma, such as 1,3,244,244 for only one input;\n"
      "    1,3,224,224:1,5 for two inputs) type: string\n"
      "    default: 1,3,224,224 \n"
      "  --input_data_path (The path of input image, if not set\n"
      "    input_data_path, the input will be 1.0.) type: string \n "
      "  --threads (Threads num) type: int32 default: 1 \n"
      "  --power_mode (Arm power mode: 0 for big cluster, 1 for little\n"
      "    cluster, 2 for all cores, 3 for no bind) type: int32 default: 3\n"
      "  --warmup (Warmup times) type: int32 default: 0 \n"
      "  --repeats (Repeats times) type: int32 default: 1 \n"
      "  --result_path (Save the inference time to the file.) type: \n"
      "    string default: result.txt \n"
      "  --use_fp16 (opening use_fp16 when run fp16 model) type: bool default: "
      "false"
      "Note that: \n"
      "  If load the optimized model, set optimized_model_path. Otherwise, \n"
      "    set model_dir, model_filename and params_filename according to \n"
      "    the fact. \n"
      "For example: \n"
      "     ./benchmark_bin --model_dir=./tmp/ --model_filename=model \n"
      "                     --params_filename=params \n";

  LOG(INFO) << help_info;
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

void Run(const std::string& model_file,
         const std::vector<int64_t>& input_shape) {
  lite::Timer timer;
  std::vector<float> perf_vct;

  // set config and create predictor
  timer.Start();
  lite_api::MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_threads(FLAGS_threads);
  config.set_power_mode(static_cast<PowerMode>(FLAGS_power_mode));

  // set backend config info
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
    // CL_PRECISION_AUTO: 0, first fp16 if valid, default
    // CL_PRECISION_FP32: 1, force fp32
    // CL_PRECISION_FP16: 2, force fp16
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

  auto predictor = lite_api::CreatePaddlePredictor(config);
  float init_time = timer.Stop();

  // set input
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(input_shape);
  auto input_data = input_tensor->mutable_data<float>();
  int64_t input_num = ShapeProduction(input_shape);
  if (FLAGS_input_data_path.empty()) {
    for (int i = 0; i < input_num; ++i) {
      input_data[i] = 1.f;
    }
  } else {
    std::fstream fs(FLAGS_input_data_path);
    if (!fs.is_open()) {
      LOG(FATAL) << "open input image " << FLAGS_input_data_path << " error.";
    }
    for (int i = 0; i < input_num; i++) {
      fs >> input_data[i];
    }
  }

  // warmup
  for (int i = 0; i < FLAGS_warmup; ++i) {
    if (i == 0) {
      timer.Start();
      predictor->Run();
      perf_vct.push_back(timer.Stop());
    } else {
      predictor->Run();
    }
  }

  // run
  for (int i = 0; i < FLAGS_repeats; ++i) {
    timer.Start();
    predictor->Run();
    perf_vct.push_back(timer.Stop());
  }

  // get output
  size_t output_tensor_num = predictor->GetOutputNames().size();
  STL::stringstream out_ss;
  out_ss << "output tensor num: " << output_tensor_num;

  for (auto tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor->GetOutput(tidx);
    out_ss << "\n--- output tensor " << tidx << " ---\n";
    auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, ShapeProduction(out_shape));
    auto out_std_dev = compute_standard_deviation<float>(
        out_data, ShapeProduction(out_shape), true, out_mean);

    out_ss << "output shape(NCHW): " << ShapePrint(out_shape) << std::endl;
    out_ss << "output tensor " << tidx
           << " elem num: " << ShapeProduction(out_shape) << std::endl;
    out_ss << "output tensor " << tidx << " mean value: " << out_mean
           << std::endl;
    out_ss << "output tensor " << tidx << " standard deviation: " << out_std_dev
           << std::endl;

    // print output
    if (FLAGS_show_output_elem) {
      for (int i = 0; i < ShapeProduction(out_shape); ++i) {
        out_ss << "out[" << tidx << "][" << i
               << "]:" << output_tensor->data<float>()[i] << std::endl;
      }
    }
  }

  // save benchmark info
  float first_time = perf_vct[0];
  if (FLAGS_warmup > 0) {
    perf_vct.erase(perf_vct.cbegin());
  }
  std::stable_sort(perf_vct.begin(), perf_vct.end());
  float perf_avg =
      std::accumulate(perf_vct.begin(), perf_vct.end(), 0.0) / FLAGS_repeats;

  STL::stringstream ss;
  ss.precision(3);
#ifdef __ANDROID__
  ss << "\n======= Device Info =======\n";
  ss << get_device_info();
#endif
  ss << "\n======= Model Info =======\n";
  ss << "optimized_model_file: " << model_file << std::endl;
  ss << "input_data_path: " << FLAGS_input_data_path << std::endl;
  ss << "input_shape: " << Vector2Str(input_shape) << std::endl;
  ss << out_ss.str();
  ss << "\n======= Runtime Info =======\n";
  ss << "threads: " << FLAGS_threads << std::endl;
  ss << "power_mode: " << FLAGS_power_mode << std::endl;
  ss << "warmup: " << FLAGS_warmup << std::endl;
  ss << "repeats: " << FLAGS_repeats << std::endl;
  ss << "result_path: " << FLAGS_result_path << std::endl;
  ss << "\n======= Backend Info =======\n";
  ss << "backend: " << FLAGS_backend << std::endl;
  ss << "cpu precision: " << FLAGS_cpu_precision << std::endl;
  if (FLAGS_backend == "opencl" || FLAGS_backend == "metal" ||
      FLAGS_backend == "x86_opencl") {
    ss << "gpu precision: " << FLAGS_gpu_precision << std::endl;
    if (FLAGS_backend != "metal") {
      ss << "opencl_cache_dir: " << FLAGS_opencl_cache_dir << std::endl;
      ss << "opencl_kernel_cache_file: " << FLAGS_opencl_kernel_cache_file
         << std::endl;
      ss << "opencl_tuned_file: " << FLAGS_opencl_tuned_file << std::endl;
    }
  }
  ss << "\n======= Perf Info =======\n";
  ss << std::fixed << std::left;
  ss << "Time(unit: ms):\n";
  ss << "init  = " << std::setw(12) << init_time << std::endl;
  ss << "first = " << std::setw(12) << first_time << std::endl;
  ss << "min   = " << std::setw(12) << perf_vct.front() << std::endl;
  ss << "max   = " << std::setw(12) << perf_vct.back() << std::endl;
  ss << "avg   = " << std::setw(12) << perf_avg << std::endl;
  if (FLAGS_enable_memory_profile) {
    ss << "\nMemory Usage(unit: kB):\n";
    ss << "init  = " << std::setw(12) << init_time << std::endl;
    ss << "avg   = " << std::setw(12) << first_time << std::endl;
  }
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

int main(int argc, char** argv) {
  // Check inputs
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  bool is_opt_model =
      (FLAGS_uncombined_model_dir.empty() && FLAGS_model_file.empty() &&
       FLAGS_param_file.empty() && !FLAGS_optimized_model_path.empty());
  bool is_origin_model =
      (!FLAGS_uncombined_model_dir.empty() ||
       (!FLAGS_model_file.empty() && !FLAGS_param_file.empty()));
  if (!is_origin_model && !is_opt_model) {
    LOG(ERROR) << "No model path is set!\n";
    paddle::lite_api::PrintUsage();
    exit(0);
  }
  if (!FLAGS_uncombined_model_dir.empty() &&
      (!FLAGS_model_file.empty() && !FLAGS_param_file.empty())) {
    LOG(ERROR) << "Both --uncombined_model_dir and --model_file --param_file "
                  "are set. Only need to set one model format!";
  }

  // Get input shape
  std::vector<int64_t> input_shape =
      paddle::lite_api::GetInputShape(FLAGS_input_shape);

  // Get optimized model file
  std::string opt_model_path = FLAGS_optimized_model_path;
  if (is_origin_model) {
    if (opt_model_path.empty()) {
      opt_model_path = "opt";
    }
    paddle::lite_api::OutputOptModel(opt_model_path);
  }

  // Run
  paddle::lite_api::Run(opt_model_path + ".nb", input_shape);

  return 0;
}
