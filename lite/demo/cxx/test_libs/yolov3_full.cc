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
#include <fstream>
#include <iostream>
#include "paddle_api.h"   // NOLINT
#include "test_helper.h"  // NOLINT

DEFINE_string(model_dir,
              "",
              "the path of the model, the model and param files is under "
              "model_dir.");
DEFINE_string(model_filename,
              "",
              "the filename of model file. When the model is combined formate, "
              "please set model_file.");
DEFINE_string(param_filename,
              "",
              "the filename of param file, set param_file when the model is "
              "combined formate.");
DEFINE_string(img_path, "", "the path of input image");
DEFINE_string(img_txt_path,
              "",
              "the path of input image, the image is processed "
              " and saved in txt file");
DEFINE_string(out_values,
              "",
              "The output values, separated by colon and comma");
DEFINE_double(threshold,
              1e-3,
              "If the output value diff is smaller than threshold, pass test");

void OptModel(const std::string& load_model_dir,
              const std::string& model_filename,
              const std::string& params_filename,
              const std::string& save_model_path) {
  paddle::lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
  if (!model_filename.empty() && !params_filename.empty()) {
    config.set_model_file(load_model_dir + "/" + model_filename);
    config.set_param_file(load_model_dir + "/" + params_filename);
  }
  std::vector<paddle::lite_api::Place> vaild_places = {
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt32)},
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt64)},
  };
  config.set_valid_places(vaild_places);

  auto predictor = paddle::lite_api::CreatePaddlePredictor(config);

  std::string cmd_str = "rm -rf " + save_model_path;
  int ret = system(cmd_str.c_str());
  if (ret == 0) {
    std::cout << "Delete old optimized model " << save_model_path << std::endl;
  }
  predictor->SaveOptimizedModel(save_model_path,
                                paddle::lite_api::LiteModelType::kNaiveBuffer);
  std::cout << "Load model from " << load_model_dir << std::endl;
  std::cout << "Save optimized model to " << save_model_path << std::endl;
}

void Run(const std::string& model_path,
         const std::string& img_path,
         const std::string& img_txt_path,
         const std::vector<double>& out_values,
         const float threshold,
         const int height,
         const int width) {
  // set config and create predictor
  paddle::lite_api::MobileConfig config;
  config.set_threads(3);
  config.set_model_from_file(model_path);

  auto predictor = paddle::lite_api::CreatePaddlePredictor(config);

  // set input
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize({1, 3, height, width});
  auto input_data = input_tensor->mutable_data<float>();
  if (img_txt_path.size() > 0) {
    std::fstream fs(img_txt_path);
    if (!fs.is_open()) {
      std::cerr << "Fail to open img txt file:" << img_txt_path << std::endl;
    }
    int num = 1 * 3 * height * width;
    for (int i = 0; i < num; i++) {
      fs >> input_data[i];
    }
  } else {
    cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "Fail to open img:" << img_path << std::endl;
      exit(1);
    }
    float means[3] = {0.485f, 0.456f, 0.406f};
    float scales[3] = {0.229f, 0.224f, 0.225f};
    process_img(img, width, height, input_data, means, scales);
  }
  auto shape_tensor = predictor->GetInput(1);
  shape_tensor->Resize({1, 2});
  auto* shape_data = shape_tensor->mutable_data<int>();
  shape_data[0] = height;
  shape_data[1] = width;

  predictor->Run();

  auto out_tensor = predictor->GetOutput(0);
  auto* out_data = out_tensor->data<float>();
  int64_t output_num = ShapeProduction(out_tensor->shape());
  bool is_pass = true;
  for (int i = 0; i < output_num && i < out_values.size(); i++) {
    std::cout << "id:" << i << " out_data:" << out_data[i]
              << " gt_data:" << out_values[i] << std::endl;
    if (fabs(out_data[i] - out_values[i]) > threshold) {
      is_pass = false;
    }
  }
  if (is_pass) {
    std::cout << "----------Pass test---------- \n\n";
  } else {
    std::cout << "----------Fail test---------- \n\n";
  }
}

int main(int argc, char** argv) {
  // Check inputs
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty() ||
      (FLAGS_img_path.empty() && FLAGS_img_txt_path.empty())) {
    std::cerr << "Input error." << std::endl;
    std::cerr
        << "Usage: " << argv[0] << std::endl
        << "--model_dir: the path of not optimized model \n"
           "--model_filename: the model filename of not optimized model \n"
           "--param_filename: the param filename of not optimized model \n"
           "--img_txt_path: the path of input image, the image is processed \n"
           "  and saved in txt file \n"
           "--img_path: the path of input image \n"
           "--out_values: The output values, separated by colon and comma.\n"
           "--threshold: If the out value diff is smaller than threshold,\n"
           "  pass test. Default 1e-3.\n";
    exit(1);
  }

  const int height = 608;
  const int width = 608;
  std::vector<double> out_values = GetDoubleNumsFromStr(FLAGS_out_values);

  std::string model_dir = FLAGS_model_dir;
  if (model_dir.back() == '/') {
    model_dir.pop_back();
  }
  std::string optimized_model_path = model_dir + "_opt2";
  OptModel(FLAGS_model_dir,
           FLAGS_model_filename,
           FLAGS_param_filename,
           optimized_model_path);
  std::string run_model_path = optimized_model_path + ".nb";

  // Run test
  Run(run_model_path,
      FLAGS_img_path,
      FLAGS_img_txt_path,
      out_values,
      FLAGS_threshold,
      height,
      width);
  return 0;
}
