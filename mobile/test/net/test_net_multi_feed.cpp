/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef PADDLE_MOBILE_CL

#include <fstream>
#include <iostream>
#include <string>
#include "../test_helper.h"
#include "../test_include.h"

void test(int argc, char *argv[]);

void feed(PaddleMobile<paddle_mobile::GPU_CL> *paddle_mobile, const DDim &dims,
          std::string feed_name) {
  float *input_data_array = new float[product(dims)];
  std::ifstream in(feed_name, std::ios::in);
  for (int i = 0; i < product(dims); i++) {
    float num;
    in >> num;
    input_data_array[i] = num;
  }
  in.close();
  framework::Tensor input_tensor(input_data_array, dims);
  DLOG << feed_name << " : " << input_tensor;
  paddle_mobile->Feed(feed_name, input_tensor);
}
int main(int argc, char *argv[]) {
  test(argc, argv);
  return 0;
}

void test(int argc, char *argv[]) {
  int arg_index = 1;
  bool fuse = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  bool enable_memory_optimization = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  bool quantification = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  int quantification_fold = std::stoi(argv[arg_index]);
  arg_index++;
  paddle_mobile::PaddleMobileConfigInternal config;
  config.memory_optimization_level = enable_memory_optimization
                                         ? MemoryOptimizationWithoutFeeds
                                         : NoMemoryOptimization;

#ifdef PADDLE_MOBILE_CL
  //  config.load_when_predict = true;
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile(config);
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
  std::cout << "testing opencl yyz " << std::endl;
#else
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile(config);
  paddle_mobile.SetThreadNum(1);
  std::cout << "testing cpu yyz " << std::endl;
#endif

  int dim_count = std::stoi(argv[arg_index]);
  arg_index++;
  int size = 1;

  arg_index += dim_count;

  bool is_lod = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  paddle_mobile::framework::LoD lod{{}};
  if (is_lod) {
    int lod_count = std::stoi(argv[arg_index]);
    arg_index++;
    for (int i = 0; i < lod_count; i++) {
      int dim = std::stoi(argv[arg_index + i]);
      lod[0].push_back(dim);
    }
    arg_index += lod_count;
  }

  int var_count = std::stoi(argv[arg_index]);
  arg_index++;
  bool is_sample_step = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  int sample_arg = std::stoi(argv[arg_index]);
  int sample_step = sample_arg;
  int sample_num = sample_arg;
  arg_index++;
  std::vector<std::string> var_names;
  for (int i = 0; i < var_count; i++) {
    std::string var_name = argv[arg_index + i];
    var_names.push_back(var_name);
  }
  arg_index += var_count;
  bool check_shape = std::stoi(argv[arg_index]) == 1;
  arg_index++;

  auto time1 = time();
  if (paddle_mobile.Load("./checked_model/model", "./checked_model/params",
                         fuse, quantification, 1, is_lod,
                         quantification_fold)) {
    auto time2 = time();
    std::cout << "auto-test"
              << " load-time-cost :" << time_diff(time1, time2) << "ms"
              << std::endl;

    feed(&paddle_mobile, {1, 4, 256, 288}, "input_rgb");
    feed(&paddle_mobile, {1, 128, 64, 72}, "last_input");
    feed(&paddle_mobile, {1, 64, 72, 2}, "grid");
    feed(&paddle_mobile, {1, 1, 64, 72}, "reliable");
    paddle_mobile.Predict();

#ifdef PADDLE_MOBILE_CL
    for (auto var_name : var_names) {
      auto cl_image = paddle_mobile.FetchImage(var_name);
      if (cl_image == nullptr || cl_image->GetCLImage() == nullptr) {
        continue;
      }
      auto len = cl_image->numel();
      if (len == 0) {
        continue;
      }
      size_t width = cl_image->ImageDims()[0];
      size_t height = cl_image->ImageDims()[1];
      paddle_mobile::framework::half_t *image_data =
          new paddle_mobile::framework::half_t[height * width * 4];
      cl_int err;
      cl_mem image = cl_image->GetCLImage();
      size_t origin[3] = {0, 0, 0};
      size_t region[3] = {width, height, 1};
      err = clEnqueueReadImage(cl_image->CommandQueue(), image, CL_TRUE, origin,
                               region, 0, 0, image_data, 0, NULL, NULL);
      CL_CHECK_ERRORS(err);
      float *tensor_data = new float[cl_image->numel()];
      auto converter = cl_image->Converter();
      converter->ImageToNCHW(image_data, tensor_data, cl_image->ImageDims(),
                             cl_image->dims());

      auto data = tensor_data;
      std::string sample = "";
      if (check_shape) {
        for (int i = 0; i < cl_image->dims().size(); i++) {
          sample += " " + std::to_string(cl_image->dims()[i]);
        }
      }
      if (!is_sample_step) {
        sample_step = len / sample_num;
      }
      if (sample_step <= 0) {
        sample_step = 1;
      }
      for (int i = 0; i < len; i += sample_step) {
        sample += " " + std::to_string(data[i]);
      }
      std::cout << "auto-test"
                << " var " << var_name << sample << std::endl;
    }
#else
    for (auto var_name : var_names) {
      auto out = paddle_mobile.Fetch(var_name);
      auto len = out->numel();
      if (len == 0) {
        continue;
      }
      if (out->memory_size() == 0) {
        continue;
      }
      if (out->type() == type_id<int>()) {
        auto data = out->data<int>();
        std::string sample = "";
        if (check_shape) {
          for (int i = 0; i < out->dims().size(); i++) {
            sample += " " + std::to_string(out->dims()[i]);
          }
        }
        if (!is_sample_step) {
          sample_step = len / sample_num;
        }
        if (sample_step <= 0) {
          sample_step = 1;
        }
        for (int i = 0; i < len; i += sample_step) {
          sample += " " + std::to_string(data[i]);
        }
        std::cout << "auto-test"
                  << " var " << var_name << sample << std::endl;
      } else if (out->type() == type_id<float>()) {
        auto data = out->data<float>();
        std::string sample = "";
        if (check_shape) {
          for (int i = 0; i < out->dims().size(); i++) {
            sample += " " + std::to_string(out->dims()[i]);
          }
        }
        if (!is_sample_step) {
          sample_step = len / sample_num;
        }
        if (sample_step <= 0) {
          sample_step = 1;
        }
        for (int i = 0; i < len; i += sample_step) {
          sample += " " + std::to_string(data[i]);
        }
        std::cout << "auto-test"
                  << " var " << var_name << sample << std::endl;
      }
    }
#endif
    std::cout << std::endl;
  }
}
#else
int main() {}
#endif
