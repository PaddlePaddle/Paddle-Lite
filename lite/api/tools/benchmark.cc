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

#include "lite/api/tools/benchmark.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "lite/core/version.h"
#include "lite/utils/timer.h"

int main(int argc, char* argv[]) {
  return paddle::lite_api::Benchmark(argc, argv);
}

namespace paddle {
namespace lite_api {

int Benchmark(int argc, char** argv) {
  gflags::SetVersionString(lite::version());
  gflags::SetUsageMessage(PrintUsage());
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Check flags validation
  bool is_opt_model =
      (FLAGS_uncombined_model_dir.empty() && FLAGS_model_file.empty() &&
       FLAGS_param_file.empty() && !FLAGS_optimized_model_path.empty());
  bool is_origin_model =
      (!FLAGS_uncombined_model_dir.empty() ||
       (!FLAGS_model_file.empty() && !FLAGS_param_file.empty()));
  if (!is_origin_model && !is_opt_model) {
    std::cerr << "\nNo model path is set!" << std::endl;
    gflags::ProgramUsage();
    exit(0);
  }
  if (!FLAGS_uncombined_model_dir.empty() &&
      (!FLAGS_model_file.empty() && !FLAGS_param_file.empty())) {
    std::cerr << "Both --uncombined_model_dir and --model_file --param_file "
                 "are set. Only need to set one model format!"
              << std::endl;
    gflags::ProgramUsage();
    exit(0);
  }

  // Get input shape
  std::vector<int64_t> input_shape = GetInputShape(FLAGS_input_shape);

  // Get optimized model file if necessary
  std::string opt_model_path = FLAGS_optimized_model_path;
  if (is_origin_model) {
    if (opt_model_path.empty()) {
      opt_model_path = "opt";
    }
    OutputOptModel(opt_model_path);
  }

  // Run
  Run(opt_model_path + ".nb", input_shape);

  return 0;
}

void Run(const std::string& model_file,
         const std::vector<int64_t>& input_shape) {
  lite::Timer timer;
  std::vector<float> perf_vct;

  // Set config and create predictor
  timer.Start();
  MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_threads(FLAGS_threads);
  config.set_power_mode(static_cast<PowerMode>(FLAGS_power_mode));

  // Set backend config info
  SetBackendConfig(config);

  auto predictor = CreatePaddlePredictor(config);
  float init_time = timer.Stop();

  // Set input
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
      std::cout << "Open input image " << FLAGS_input_data_path << " error."
                << std::endl;
    }
    for (int i = 0; i < input_num; i++) {
      fs >> input_data[i];
    }
  }

  // Warmup
  for (int i = 0; i < FLAGS_warmup; ++i) {
    if (i == 0) {
      timer.Start();
      predictor->Run();
      perf_vct.push_back(timer.Stop());
    } else {
      predictor->Run();
    }
    timer.SleepInMs(FLAGS_run_delay);
  }

  // Run
  for (int i = 0; i < FLAGS_repeats; ++i) {
    timer.Start();
    predictor->Run();
    perf_vct.push_back(timer.Stop());
    timer.SleepInMs(FLAGS_run_delay);
  }

  // Get output
  size_t output_tensor_num = predictor->GetOutputNames().size();
  STL::stringstream out_ss;
  out_ss << "output tensor num: " << output_tensor_num;

  for (auto tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const Tensor> output_tensor = predictor->GetOutput(tidx);
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

    if (FLAGS_show_output_elem) {
      for (int i = 0; i < ShapeProduction(out_shape); ++i) {
        out_ss << "out[" << tidx << "][" << i
               << "]:" << output_tensor->data<float>()[i] << std::endl;
      }
    }
  }

  // Save benchmark info
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
  ss << "input_data_path: "
     << (FLAGS_input_data_path.empty() ? "All 1.f" : FLAGS_input_data_path)
     << std::endl;
  ss << "input_shape: " << Vector2Str(input_shape) << std::endl;
  ss << out_ss.str();
  ss << "\n======= Runtime Info =======\n";
  ss << "benchmark_bin version: " << lite::version() << std::endl;
  ss << "threads: " << FLAGS_threads << std::endl;
  ss << "power_mode: " << FLAGS_power_mode << std::endl;
  ss << "warmup: " << FLAGS_warmup << std::endl;
  ss << "repeats: " << FLAGS_repeats << std::endl;
  if (FLAGS_run_delay > 0.f) {
    ss << "run_delay(sec): " << FLAGS_run_delay << std::endl;
  }
  ss << "result_path: " << FLAGS_result_path << std::endl;
  ss << "\n======= Backend Info =======\n";
  ss << "backend: " << FLAGS_backend << std::endl;
  ss << "cpu precision: " << FLAGS_cpu_precision << std::endl;
  if (FLAGS_backend == "opencl" || FLAGS_backend == "x86_opencl") {
    ss << "gpu precision: " << FLAGS_gpu_precision << std::endl;
    ss << "opencl_cache_dir: " << FLAGS_opencl_cache_dir << std::endl;
    ss << "opencl_kernel_cache_file: " << FLAGS_opencl_kernel_cache_file
       << std::endl;
    ss << "opencl_tuned_file: " << FLAGS_opencl_tuned_file << std::endl;
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
    ss << "init  = " << std::setw(12) << "Not supported yet" << std::endl;
    ss << "avg   = " << std::setw(12) << "Not supported yet" << std::endl;
  }
  std::cout << ss.str() << std::endl;
  std::ofstream ofs(FLAGS_result_path, std::ios::app);
  if (!ofs.is_open()) {
    std::cout << "Fail to open result file: " << FLAGS_result_path << std::endl;
  }
  ofs << ss.str();
  ofs.close();
}

}  // namespace lite_api
}  // namespace paddle
