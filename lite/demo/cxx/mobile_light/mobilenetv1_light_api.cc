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

#include <sys/time.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
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

std::vector<std::string> split_string(const std::string& str_in) {
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
}

std::vector<int64_t> get_shape(const std::string& str_shape) {
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

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void RunModel(std::string model_dir,
              const std::vector<shape_t>& input_shapes,
              size_t repeats,
              size_t warmup,
              size_t power_mode,
              size_t thread_num,
              size_t accelerate_opencl,
              size_t print_output_elem) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_dir);

#ifdef METAL
  std::string metal_lib_path = "../../../metal/lite.metallib";
  config.set_metal_lib_path(metal_lib_path);
  config.set_metal_use_mps(true);
#else
  // NOTE: Use android gpu with opencl, you should ensure:
  //  first, [compile **cpu+opencl** paddlelite
  //    lib](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/opencl.md);
  //  second, [convert and use opencl nb
  //    model](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/user_guides/opt/opt_bin.md).

  bool is_opencl_backend_valid =
      ::IsOpenCLBackendValid(/*check_fp16_valid = false*/);
  std::cout << "is_opencl_backend_valid:"
            << (is_opencl_backend_valid ? "true" : "false") << std::endl;
  if (is_opencl_backend_valid) {
    if (accelerate_opencl != 0) {
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
      const std::string bin_path = "/data/local/tmp/";
      const std::string bin_name = "lite_opencl_kernel.bin";
      config.set_opencl_binary_path_name(bin_path, bin_name);

      // opencl tune option
      // CL_TUNE_NONE: 0
      // CL_TUNE_RAPID: 1
      // CL_TUNE_NORMAL: 2
      // CL_TUNE_EXHAUSTIVE: 3
      const std::string tuned_path = "/data/local/tmp/";
      const std::string tuned_name = "lite_opencl_tuned.bin";
      config.set_opencl_tune(CL_TUNE_NORMAL, tuned_path, tuned_name);

      // opencl precision option
      // CL_PRECISION_AUTO: 0, first fp16 if valid, default
      // CL_PRECISION_FP32: 1, force fp32
      // CL_PRECISION_FP16: 2, force fp16
      config.set_opencl_precision(CL_PRECISION_FP16);
    }
  } else {
    std::cout << "*** nb model will be running on cpu. ***" << std::endl;
    // you can give backup cpu nb model instead
    // config.set_model_from_file(cpu_nb_model_dir);
  }
#endif

  // NOTE: To load model transformed by model_optimize_tool before
  // release/v2.3.0, plese use `set_model_dir` API as listed below.
  // config.set_model_dir(model_dir);
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  config.set_threads(thread_num);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data
  std::cout << "input_shapes.size():" << input_shapes.size() << std::endl;
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

  // 4. Run predictor
  double first_duration{-1};
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      auto start = GetCurrentUS();
      predictor->Run();
      first_duration = (GetCurrentUS() - start) / 1000.0;
    } else {
      predictor->Run();
    }
  }

  double sum_duration = 0.0;  // millisecond;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    auto start = GetCurrentUS();

    predictor->Run();

    auto duration = (GetCurrentUS() - start) / 1000.0;
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    std::cout << "run_idx:" << ridx + 1 << " / " << repeats << ": " << duration
              << " ms" << std::endl;
    if (first_duration < 0) {
      first_duration = duration;
    }
  }
  avg_duration = sum_duration / static_cast<float>(repeats);
  std::cout << "\n======= benchmark summary =======\n"
            << "input_shape(s) (NCHW):" << ShapePrint(input_shapes) << "\n"
            << "model_dir:" << model_dir << "\n"
            << "warmup:" << warmup << "\n"
            << "repeats:" << repeats << "\n"
            << "power_mode:" << power_mode << "\n"
            << "thread_num:" << thread_num << "\n"
            << "*** time info(ms) ***\n"
            << "1st_duration:" << first_duration << "\n"
            << "max_duration:" << max_duration << "\n"
            << "min_duration:" << min_duration << "\n"
            << "avg_duration:" << avg_duration << "\n";

  // 5. Get output
  std::cout << "\n====== output summary ====== " << std::endl;
  size_t output_tensor_num = predictor->GetOutputNames().size();
  std::cout << "output tensor num:" << output_tensor_num << std::endl;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor->GetOutput(tidx);
    std::cout << "\n--- output tensor " << tidx << " ---" << std::endl;
    auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, ShapeProduction(out_shape));
    auto out_std_dev = compute_standard_deviation<float>(
        out_data, ShapeProduction(out_shape), true, out_mean);

    std::cout << "output shape(NCHW):" << ShapePrint(out_shape) << std::endl;
    std::cout << "output tensor " << tidx
              << " elem num:" << ShapeProduction(out_shape) << std::endl;
    std::cout << "output tensor " << tidx
              << " standard deviation:" << out_std_dev << std::endl;
    std::cout << "output tensor " << tidx << " mean value:" << out_mean
              << std::endl;

    // print output
    if (print_output_elem) {
      for (int i = 0; i < ShapeProduction(out_shape); ++i) {
        std::cout << "out[" << tidx << "][" << i
                  << "]:" << output_tensor->data<float>()[i] << std::endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  std::vector<std::string> str_input_shapes;
  std::vector<shape_t> input_shapes{
      {1, 3, 224, 224}};  // shape_t ==> std::vector<int64_t>

  int repeats = 10;
  int warmup = 10;
  // set arm power mode:
  // 0 for big cluster, high performance
  // 1 for little cluster
  // 2 for all cores
  // 3 for no bind
  size_t power_mode = 0;
  size_t thread_num = 1;
  int accelerate_opencl = 1;
  int print_output_elem = 0;

  if (argc > 2 && argc < 9) {
    std::cerr
        << "usage: ./" << argv[0] << "\n"
        << "  <naive_buffer_model_dir>\n"
        << "  <raw_input_shapes>, eg: 1,3,224,224 for 1 input; "
           "1,3,224,224:1,5 for 2 inputs\n"
        << "  <repeats>, eg: 100\n"
        << "  <warmup>, eg: 10\n"
        << "  <power_mode>, 0: big cluster, high performance\n"
           "                1: little cluster\n"
           "                2: all cores\n"
           "                3: no bind\n"
        << "  <thread_num>, eg: 1 for single thread \n"
        << "  <accelerate_opencl>, this option takes effect only when model "
           "can be running on opencl backend.\n"
           "                       0: disable opencl kernel cache & tuning\n"
           "                       1: enable opencl kernel cache & tuning\n"
        << "  <print_output>, 0: disable print outputs to stdout\n"
           "                  1: enable print outputs to stdout\n"
        << std::endl;
    return 0;
  }

  std::string model_dir = argv[1];
  if (argc >= 9) {
    input_shapes.clear();
    std::string raw_input_shapes = argv[2];
    std::cout << "raw_input_shapes: " << raw_input_shapes << std::endl;
    str_input_shapes = split_string(raw_input_shapes);
    for (size_t i = 0; i < str_input_shapes.size(); ++i) {
      std::cout << "input shape: " << str_input_shapes[i] << std::endl;
      input_shapes.push_back(get_shape(str_input_shapes[i]));
    }

    repeats = atoi(argv[3]);
    warmup = atoi(argv[4]);
    power_mode = atoi(argv[5]);
    thread_num = atoi(argv[6]);
    accelerate_opencl = atoi(argv[7]);
    print_output_elem = atoi(argv[8]);
  }

  RunModel(model_dir,
           input_shapes,
           repeats,
           warmup,
           power_mode,
           thread_num,
           accelerate_opencl,
           print_output_elem);

  return 0;
}
