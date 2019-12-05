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

#include <dirent.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"


DEFINE_string(input_file, "", "input_file");

namespace paddle {
namespace lite {

std::vector<std::string> GetDirectoryFiles(const std::string& dir) {
  std::vector<std::string> files;
  std::shared_ptr<DIR> directory_ptr(opendir(dir.c_str()),
                                     [](DIR* dir) { dir&& closedir(dir); });
  struct dirent* dirent_ptr;
  if (!directory_ptr) {
    std::cout << "Error opening : " << std::strerror(errno) << dir << std::endl;
    return files;
  }

  while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
    files.push_back(std::string(dirent_ptr->d_name));
  }
  return files;
}

void readFromFile(int num, std::string path, float* data) {
  std::ifstream file_stream(path);
  // file_stream.open(path);
  if (!file_stream.good()) {
    std::cout << "file: " << path << " dones not exist!\n";
    exit(-1);
    return;
  }
  // float* data = mutableData<float>();
  for (int i = 0; i < num; ++i) {
    float value = 0;
    file_stream >> value;
    data[i] = value;
  }
  file_stream.close();
}

// #ifdef LITE_WITH_FPGA
TEST(ResNet50, test) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({
      Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  predictor.Build(FLAGS_model_dir,
                  "",
                  "",
                  valid_places);


  // predictor.Build(FLAGS_model_dir,
  //                 FLAGS_model_dir + "/model",
  //                 FLAGS_model_dir + "/params",
  //                 Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
  //                 valid_places);


  auto* input_tensor = predictor.GetInput(0);

  int width = 416;
  int height = 416;


  std::ifstream file_stream(FLAGS_input_file);
  // file_stream.open(path);
  if (!file_stream.good()) {
    std::cout << "file: " << FLAGS_input_file << " dones not exist!\n";
    exit(-1);
    return;
  }

  file_stream >> height;
  file_stream >> width;

  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, height, width})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();

  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  // readFromFile(item_size, "car.data", data);

  int num = 3 * width * height;

  for (int i = 0; i < num; ++i) {
    float value = 0;
    file_stream >> value;
    data[i] = value;
  }
  file_stream.close();

  for (int i = 0; i < 2; ++i) {
    predictor.Run();
  }

  auto* out = predictor.GetOutput(0);
  for (int i = 0;i < out->dims().production();i++) {
    std::cout << ":" << out->data<float>()[i] << std::endl; 
  }

  // std::cout << "-------\n";
  // auto* out1 = predictor.GetOutput(1);
  // for (int i = 0;i < out1->dims().production();i++) {
  //   std::cout << ":" << out1->data<float>()[i] << std::endl; 
  // }

  // std::string file = "output/" + FLAGS_input_file.substr (6);
  // std::cout << "file:::" << file << std::endl;

  // std::ofstream ofs;
  // ofs.open(file);
  // for (int i = 0; i < out->dims().production(); i++) {
  //   float value = out->data<float>()[i];
  //   ofs << value << std::endl;
  // }
  // ofs.close();

  LOG(INFO) << "================== Speed Report ===================";
}
// #endif

}  // namespace lite
}  // namespace paddle
