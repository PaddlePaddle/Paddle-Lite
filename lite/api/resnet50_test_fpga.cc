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

#ifdef LITE_WITH_FPGA
TEST(ResNet50, test) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({
      Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNHWC)},
      // Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNHWC)},
  });

  predictor.Build(FLAGS_model_dir,
                  "",
                  "",
                  Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 1000, 1000})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();

  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }
  for (int i = 0; i < 1; ++i) {
    predictor.Run();
  }

  // std::string path = "inputs";
  // auto files = GetDirectoryFiles(path);
  // for (auto p : files) {
  //   std::string pp = path + "/" + p;
  //   std::cout << "\n path::::========== " << pp << std::endl;
  //   std::size_t found = pp.find(".txt");
  //   if (found == std::string::npos) {
  //     continue;
  //   }
  //   // readFromFile(item_size, pp, data);
  //   predictor.Run();

  //   auto* output_tensor = predictor.GetOutput(0);
  //   lite::Tensor* out = const_cast<lite::Tensor*>(output_tensor);
  //   auto* out_data = out->data<float>();
  //   item_size = out->dims().production();

  //   float max = 0;
  //   int index = 0;
  //   for (int i = 0; i < item_size; i++) {
  //     float value = data[i];
  //     if (value > max) {
  //       max = value;
  //       index = i;
  //     }
  //     std::cout << i << " : " << value << std::endl;
  //   }
  //   std::cout << "max:" << max << " @ :" << index << std::endl;
  //   std::cout << "size:" << predictor.GetOutputs()->size() << std::endl;

  //   std::cout << "output_tensor:::" << out << std::endl;
  //   std::cout << "out_data:::" << out_data << std::endl;
  // }

  LOG(INFO) << "================== Speed Report ===================";
}
#endif

}  // namespace lite
}  // namespace paddle
