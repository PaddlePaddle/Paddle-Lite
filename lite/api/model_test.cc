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

#include <sstream>
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
#endif  // LITE_WITH_PROFILE
#include <gflags/gflags.h>

using paddle::lite::profile::Timer;

DEFINE_string(input_shape,
              "1,3,224,224",
              "input shapes, separated by colon and comma");
DEFINE_bool(use_optimize_nb,
            false,
            "optimized & naive buffer model for mobile devices");
DEFINE_string(backend,
              "arm_cpu",
              "choose backend for valid_places: arm_cpu | opencl. Compile "
              "OpenCL version if you choose opencl");
DEFINE_string(arg_name, "", "the arg name");
DEFINE_string(in_txt, "", "input text");
DEFINE_string(out_txt, "", "output text");

namespace paddle {
namespace lite_api {

void OutputOptModel(const std::string& load_model_dir,
                    const std::string& save_optimized_model_dir,
                    const std::vector<std::vector<int64_t>>& input_shapes) {
  lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
#ifdef LITE_WITH_X86
  config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                           Place{TARGET(kX86), PRECISION(kInt64)},
                           Place{TARGET(kHost), PRECISION(kFloat)}});
#else
  if (FLAGS_backend == "opencl") {
    config.set_valid_places({
        Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},
        Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},
        TARGET(kARM),  // enable kARM CPU kernel when no opencl kernel
    });
  } else {  // arm_cpu
    config.set_valid_places({
        Place{TARGET(kARM), PRECISION(kFloat)},
    });
  }
#endif
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
         const int warmup_times = 0) {
  lite_api::MobileConfig config;
  config.set_model_from_file(model_dir + ".nb");
  config.set_power_mode(power_mode);
  config.set_threads(thread_num);

  auto predictor = lite_api::CreatePaddlePredictor(config);
  bool flag_in = true;
  bool flag_out = true;
  if (FLAGS_in_txt == "") {
    flag_in = false;
  }
  if (FLAGS_out_txt == "") {
    flag_out = false;
  }
  printf("flag_in: %d, flag_out: %d \n", flag_in, flag_out);

  for (int j = 0; j < input_shapes.size(); ++j) {
    auto input_tensor = predictor->GetInput(j);
    input_tensor->Resize(input_shapes[j]);
    auto input_data = input_tensor->mutable_data<float>();
    int input_num = 1;
    for (int i = 0; i < input_shapes[j].size(); ++i) {
      input_num *= input_shapes[j][i];
    }
    FILE* fp_r = nullptr;
    if (flag_in) {
      fp_r = fopen(FLAGS_in_txt.c_str(), "r");
    }
    for (int i = 0; i < input_num; ++i) {
      if (flag_in) {
        fscanf(fp_r, "%f\n", &input_data[i]);
      } else {
        input_data[i] = 1.f;
      }
    }
    if (flag_in) {
      fclose(fp_r);
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
    LOG(INFO) << "iter: " << j << ", time: " << t << " ms";
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir
            << ", power_mode: " << static_cast<int>(power_mode)
            << ", threads num " << thread_num << ", warmup: " << warmup_times
            << ", repeats: " << repeat << ", avg time: " << ti.LapTimes().Avg()
            << " ms"
            << ", min time: " << ti.LapTimes().Min() << " ms"
            << ", max time: " << ti.LapTimes().Max() << " ms.";

  // output summary
  size_t output_tensor_num = predictor->GetOutputNames().size();
  LOG(INFO) << "output tensor num:" << output_tensor_num;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    auto output_tensor = predictor->GetOutput(tidx);
    LOG(INFO) << "============= output tensor " << tidx << " =============";
    auto tensor_shape = output_tensor->shape();
    std::string tensor_shape_str{""};
    int output_tensor_numel = 1;
    for (int i = 0; i < tensor_shape.size(); ++i) {
      output_tensor_numel *= tensor_shape[i];
      tensor_shape_str += std::to_string(tensor_shape[i]);
      tensor_shape_str += (i < tensor_shape.size() - 1) ? "x" : "";
    }
    auto out_data = output_tensor->data<float>();
    auto out_mean =
        paddle::lite::compute_mean<float>(out_data, output_tensor_numel);
    auto out_std_dev = paddle::lite::compute_standard_deviation<float>(
        out_data, output_tensor_numel, true, out_mean);
    FILE* fp1 = nullptr;
    if (flag_out) {
      fp1 = fopen(FLAGS_out_txt.c_str(), "w");
    }
    double sum1 = 0.f;
    for (int i = 0; i < output_tensor_numel; ++i) {
      if (flag_out) {
        fprintf(fp1, "%f\n", out_data[i]);
      }
      sum1 += out_data[i];
    }
    if (flag_out) {
      fclose(fp1);
    }
    printf("out mean: %f \n", sum1 / output_tensor_numel);

    LOG(INFO) << "output tensor " << tidx << " dims:" << tensor_shape_str;
    LOG(INFO) << "output tensor " << tidx
              << " elements num:" << output_tensor_numel;
    LOG(INFO) << "output tensor " << tidx
              << " standard deviation:" << out_std_dev;
    LOG(INFO) << "output tensor " << tidx << " mean value:" << out_mean << "\n";

    // print result
    for (int i = 0; i < output_tensor_numel; ++i) {
      VLOG(2) << "output_tensor->data<float>()[" << i
              << "]:" << output_tensor->data<float>()[i];
    }
  }

  // please turn off memory_optimize_pass to use this feature.
  if (FLAGS_arg_name != "") {
    auto arg_tensor = predictor->GetTensor(FLAGS_arg_name);
    auto arg_shape = arg_tensor->shape();
    int arg_num = 1;
    std::ostringstream os;
    os << "{";
    for (int i = 0; i < arg_shape.size(); ++i) {
      arg_num *= arg_shape[i];
      os << arg_shape[i] << ",";
    }
    os << "}";
    float sum = 0.;
    std::ofstream out(FLAGS_arg_name + ".txt");
    for (size_t i = 0; i < arg_num; ++i) {
      sum += arg_tensor->data<float>()[i];
      out << paddle::lite::to_string(arg_tensor->data<float>()[i]) << "\n";
    }
    LOG(INFO) << FLAGS_arg_name << " shape is " << os.str()
              << ", mean value is " << sum * 1. / arg_num;
  }
}
#endif

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(INFO) << "usage: "
              << "--model_dir /path/to/your/model";
    exit(0);
  }

  std::string save_optimized_model_dir = "";
  if (FLAGS_use_optimize_nb) {
    save_optimized_model_dir = FLAGS_model_dir;
  } else {
    save_optimized_model_dir = FLAGS_model_dir + "opt2";
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

  LOG(INFO) << "input shapes: " << FLAGS_input_shape;
  std::vector<std::string> str_input_shapes = split_string(FLAGS_input_shape);
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t i = 0; i < str_input_shapes.size(); ++i) {
    LOG(INFO) << "input shape: " << str_input_shapes[i];
    input_shapes.push_back(get_shape(str_input_shapes[i]));
  }

  if (!FLAGS_use_optimize_nb) {
    // Output optimized model
    paddle::lite_api::OutputOptModel(
        FLAGS_model_dir, save_optimized_model_dir, input_shapes);
  }

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  paddle::lite_api::Run(
      input_shapes,
      save_optimized_model_dir,
      static_cast<paddle::lite_api::PowerMode>(FLAGS_power_mode),
      FLAGS_threads,
      FLAGS_repeats,
      FLAGS_warmup);
#endif
  return 0;
}
