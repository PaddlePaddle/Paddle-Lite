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
#include <sstream>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/test/test_helper.h"
#include "lite/core/device_info.h"
#include "lite/core/profile/timer.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/basic_profiler.h"
#endif  // LITE_WITH_PROFILE

using paddle::lite::profile::Timer;

DEFINE_string(input_shape,
              "1,3,224,224",
              "input shapes, separated by colon and comma");
DEFINE_bool(use_optimize_nb,
            false,
            "optimized & naive buffer model for mobile devices");
DEFINE_string(arg_name, "", "the arg name");

DEFINE_string(threshold, "0.5", "threshold value default 0.5f");
DEFINE_string(in_txt, "", "input text");
DEFINE_string(out_txt, "", "output text");
DEFINE_int32(orih, 1920, "input image height");
DEFINE_int32(oriw, 1080, "input image width");

namespace paddle {
namespace lite_api {

struct Object {
  float x;
  float y;
  float width;
  float height;
  float class_id;
  float prob;
};

void OutputOptModel(const std::string& load_model_dir,
                    const std::string& save_optimized_model_dir,
                    const std::vector<std::vector<int64_t>>& input_shapes) {
  lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
  config.set_valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
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

void detect_choose(const float* dout,
                   std::vector<int64_t> dims,
                   const float thresh) {
  std::string name = FLAGS_out_txt + "_accu.txt";
  FILE* fp = fopen(name.c_str(), "w");
  for (int iw = 0; iw < dims[0]; iw++) {
    const float* values = dout + iw * dims[1];
    if (values[1] > thresh) {  // pro > 0.01
      fprintf(fp, "%f \n", values[0]);
      fprintf(fp, "%f \n", values[1]);
      fprintf(fp, "%f \n", values[2]);
      fprintf(fp, "%f \n", values[3]);
      fprintf(fp, "%f \n", values[4]);
      fprintf(fp, "%f \n", values[5]);
    }
  }
  fclose(fp);
}
void detect_object(const float* dout,
                   std::vector<int64_t> dims,
                   const float thresh,
                   int orih,
                   int oriw) {
  std::vector<Object> objects;
  for (int iw = 0; iw < dims[0]; iw++) {
    Object object;
    const float* values = dout + iw * dims[1];
    object.class_id = values[0];
    object.prob = values[1];
    object.x = values[2] * oriw;
    object.y = values[3] * orih;
    object.width = values[4] * oriw - object.x;
    object.height = values[5] * orih - object.y;
    objects.push_back(object);
  }
  std::string name = FLAGS_out_txt + "_accu.txt";
  FILE* fp = fopen(name.c_str(), "w");
  for (size_t i = 0; i < objects.size(); ++i) {
    Object object = objects.at(i);
    if (object.prob > thresh && object.x > 0 && object.y > 0 &&
        object.width > 0 && object.height > 0) {
      if (object.x >= oriw || object.width >= oriw || object.y >= orih ||
          object.height >= orih)
        continue;
      fprintf(fp, "%f \n", object.x);
      fprintf(fp, "%f \n", object.y);
      fprintf(fp, "%f \n", object.width);
      fprintf(fp, "%f \n", object.height);
      fprintf(fp, "%f \n", object.prob);
      fprintf(fp, "%f \n", object.class_id);
      LOG(INFO) << "object id: " << object.class_id << ", image size: " << oriw
                << ", " << orih << ", detect object: " << object.prob
                << ", location: x=" << object.x << ", y=" << object.y
                << ", width=" << object.width << ", height=" << object.height;
    }
  }
  fclose(fp);
}
#ifdef LITE_WITH_ARM
void Run(const std::vector<std::vector<int64_t>>& input_shapes,
         const std::string& model_dir,
         const PowerMode power_mode,
         const int thread_num,
         const int repeat,
         const int warmup_times = 0) {
  lite_api::MobileConfig config;
  config.set_model_dir(model_dir);
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

  auto output = predictor->GetOutput(0);
  auto out = output->data<float>();
  auto output_shape = output->shape();
  // detect
  detect_object(
      out, output_shape, atof(FLAGS_threshold.data()), FLAGS_orih, FLAGS_oriw);
  // detect_choose(out, output_shape, atof(FLAGS_threshold.data()));
  LOG(INFO) << "out " << out[0];
  LOG(INFO) << "out " << out[1];
  int output_num = 1;
  for (int i = 0; i < output_shape.size(); ++i) {
    output_num *= output_shape[i];
  }
  LOG(INFO) << "output_num: " << output_num;
  FILE* fp = nullptr;
  if (flag_out) {
    fp = fopen(FLAGS_out_txt.c_str(), "w");
  }
  double sum1 = 0.f;
  for (int i = 0; i < output_num; ++i) {
    if (flag_out) {
      fprintf(fp, "%f\n", out[i]);
    }
    sum1 += out[i];
  }
  if (flag_out) {
    fclose(fp);
  }

  printf("out mean: %f \n", sum1 / output_num);

  FILE* fp_w = fopen("time.txt", "a+");
  if (!fp_w) {
    printf("open file failed \n");
    return;
  }
  fprintf(fp_w,
          "model: %s, threads: %d, avg: %f ms, min: %f ms, max: %f ms \n",
          model_dir.c_str(),
          thread_num,
          ti.LapTimes().Avg(),
          ti.LapTimes().Min(),
          ti.LapTimes().Max());
  fclose(fp_w);

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

#ifdef LITE_WITH_ARM
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
