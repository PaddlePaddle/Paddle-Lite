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
  if (!CheckFlagsValid()) {
    std::cout << gflags::ProgramUsage();
    exit(0);
  }

  // Get optimized model file if necessary
  auto model_file = OutputOptModel(FLAGS_optimized_model_file);

  // Get input shapes
  auto input_shapes = lite::GetShapes(FLAGS_input_shape);

  // Run
  Run(model_file, input_shapes);

  return 0;
}

void Run(const std::string& model_file,
         const std::vector<std::vector<int64_t>>& input_shapes) {
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

  // Set inputs
  for (size_t i = 0; i < input_shapes.size(); i++) {
    auto input_tensor = predictor->GetInput(i);
    input_tensor->Resize(input_shapes[i]);
    // NOTE: Change input data type to other type as you need.
    auto input_data = input_tensor->mutable_data<float>();
    auto input_num = lite::ShapeProduction(input_shapes[i]);
    if (FLAGS_input_data_path.empty()) {
      for (auto j = 0; j < input_num; j++) {
        input_data[j] = 1.f;
      }
    } else {
      auto paths = lite::SplitString(FLAGS_input_data_path);
      std::ifstream fs(paths[i]);
      if (!fs.is_open()) {
        std::cerr << "Open input image " << paths[i] << " error." << std::endl;
      }
      for (int k = 0; k < input_num; k++) {
        fs >> input_data[k];
      }
      fs.close();
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
  std::stringstream out_ss;
  out_ss << "output tensor num: " << output_tensor_num;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const Tensor> output_tensor = predictor->GetOutput(tidx);
    out_ss << "\n--- output tensor " << tidx << " ---\n";
    auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto ele_num = lite::ShapeProduction(out_shape);
    auto out_mean = lite::compute_mean<float>(out_data, ele_num);
    auto out_std_dev = lite::compute_standard_deviation<float>(
        out_data, ele_num, true, out_mean);

    out_ss << "output shape(NCHW): " << lite::ShapePrint(out_shape)
           << std::endl;
    out_ss << "output tensor " << tidx << " elem num: " << ele_num << std::endl;
    out_ss << "output tensor " << tidx << " mean value: " << out_mean
           << std::endl;
    out_ss << "output tensor " << tidx << " standard deviation: " << out_std_dev
           << std::endl;

    if (FLAGS_show_output_elem) {
      for (int i = 0; i < ele_num; ++i) {
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

  std::stringstream ss;
  ss.precision(3);
#ifdef __ANDROID__
  ss << "\n======= Device Info =======\n";
  ss << GetDeviceInfo();
#endif
  ss << "\n======= Model Info =======\n";
  ss << "optimized_model_file: " << model_file << std::endl;
  ss << "input_data_path: "
     << (FLAGS_input_data_path.empty() ? "All 1.f" : FLAGS_input_data_path)
     << std::endl;
  ss << "input_shape: " << FLAGS_input_shape << std::endl;
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
  if (FLAGS_backend == "opencl,arm" || FLAGS_backend == "opencl" ||
      FLAGS_backend == "opencl,x86" || FLAGS_backend == "x86_opencl") {
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
  StoreBenchmarkResult(ss.str());
}

}  // namespace lite_api
}  // namespace paddle
