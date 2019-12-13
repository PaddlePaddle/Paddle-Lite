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

// float* temp_data = new float(33 * 10 * 23);

// std::vector<std::string> GetDirectoryFiles(const std::string& dir) {
//   std::vector<std::string> files;
//   std::shared_ptr<DIR> directory_ptr(opendir(dir.c_str()),
//                                      [](DIR* dir) { dir&& closedir(dir); });
//   struct dirent* dirent_ptr;
//   if (!directory_ptr) {
//     std::cout << "Error opening : " << std::strerror(errno) << dir <<
//     std::endl;
//     return files;
//   }

//   while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
//     files.push_back(std::string(dirent_ptr->d_name));
//   }
//   return files;
// }

void read_from_file(const std::string& path, float* data, int num) {
  std::ifstream file_stream;
  file_stream.open(path);
  if (!file_stream) {
    exit(-1);
    return;
  }

  for (int i = 0; i < num; ++i) {
    float value = 0;
    file_stream >> value;
    data[i] = value;
  }
}

void chw_to_hwc(float* src, float* dst, int channel, int height, int width) {
  int amount_per_row = width * channel;
  int index = 0;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      int offset_height = h * amount_per_row;
      for (int w = 0; w < width; w++) {
        int dst_index = offset_height + w * channel + c;
        dst[dst_index] = src[index];
        index = index + 1;
      }
    }
  }
}

void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place,
               bool use_npu = false) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;

  // predictor.Build(FLAGS_model_dir, "", "", preferred_place, valid_places);
  predictor.Build("", "attention/model", "attention/params", valid_places);

  auto* input_tensor = predictor.GetInput(0);
  // input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 1, 48, 512})));
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 1, 100, 200})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  // std::ifstream file_stream(FLAGS_input_file);
  // // file_stream.open(path);
  // if (!file_stream.good()) {
  //   std::cout << "file: " << FLAGS_input_file << " dones not exist!\n";
  //   exit(-1);
  //   return;
  // }

  // read_from_file("n7cu17.data", data, 100 * 200);
  read_from_file(FLAGS_input_file, data, 100 * 200);
  // read_from_file("t.data", data, 48 * 512);

  // for (int i = 0;i < 48 * 512;i++ ) {
  //   std::cout << ":" << data[i] << std::endl;
  // }

  //=============================================
  auto* init_ids = predictor.GetInput(1);
  init_ids->Resize(DDim(std::vector<DDim::value_type>({1, 1})));
  auto* data_ids = init_ids->mutable_data<float>();
  auto ids_size = init_ids->dims().production();
  for (int i = 0; i < ids_size; i++) {
    data_ids[i] = 0;
  }
  auto lod_ids = init_ids->mutable_lod();
  std::vector<std::vector<uint64_t>> lod_i{{0, 1}, {0, 1}};
  *lod_ids = lod_i;

  //=============================================
  auto* init_scores = predictor.GetInput(2);
  init_scores->Resize(DDim(std::vector<DDim::value_type>({1, 1})));
  auto* data_scores = init_scores->mutable_data<float>();
  auto scores_size = input_tensor->dims().production();
  for (int i = 0; i < scores_size; i++) {
    data_scores[i] = 0;
  }
  auto lod_scores = init_scores->mutable_lod();
  std::vector<std::vector<uint64_t>> lod_s{{0, 1}, {0, 1}};
  *lod_scores = lod_s;

  //=============================================
  auto* position_encoding = predictor.GetInput(3);
  position_encoding->Resize(
      DDim(std::vector<DDim::value_type>({1, 33, 10, 23})));
  auto* position_encoding_data = position_encoding->mutable_data<float>();

  float* temp_data = position_encoding_data;

  std::cout << "====================== 1\n";

  for (int i = 0; i < position_encoding->dims().production(); ++i) {
    temp_data[i] = 0;
  }
  std::cout << "====================== 2\n";
  int index = 0;
  for (int i = 0; i < 10; i++) {
    for (int row = 0; row < 10; row++) {
      for (int col = 0; col < 23; col++) {
        if (i == row) {
          temp_data[index] = 1.0f;
        } else {
          temp_data[index] = 0.0f;
        }
        index++;
      }
    }
  }
  std::cout << "====================== 3\n";
  for (int i = 0; i < 23; i++) {
    for (int row = 0; row < 10; row++) {
      for (int col = 0; col < 23; col++) {
        if (i == col) {
          temp_data[index] = 1.0f;
        } else {
          temp_data[index] = 0.0f;
        }
        index++;
      }
    }
  }
  std::cout << "====================== 4\n";
  // chw_to_hwc(temp_data, position_encoding_data, 33, 10, 23);
  // delete[] temp_data;

  // read_from_file("position_encoding.data", position_encoding_data, 33 * 10 *
  // 23);
  // position_encoding->ZynqTensor()->readFromFile("position_encoding.data");

  // exit(-1);

  // for (int i = 0; i < FLAGS_warmup; ++i) {
  //   predictor.Run();
  // }

  auto start = GetCurrentUS();
  for (int i = 0; i < 2; ++i) {
    predictor.Run();
  }

  std::cout << "================== Speed Report ===================";
  std::cout << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  //  std::vector<std::vector<float>> results;
  //  // i = 1
  //  results.emplace_back(std::vector<float>(
  //      {0.00019130898, 9.467885e-05,  0.00015971427, 0.0003650665,
  //       0.00026431272, 0.00060884043, 0.0002107942,  0.0015819625,
  //       0.0010323516,  0.00010079765, 0.00011006987, 0.0017364529,
  //       0.0048292773,  0.0013995157,  0.0018453331,  0.0002428986,
  //       0.00020211363, 0.00013668182, 0.0005855956,  0.00025901722}));
  auto* out = predictor.GetOutput(0);

  //  ASSERT_EQ(out->dims().size(), 2);
  //  ASSERT_EQ(out->dims()[0], 1);
  //  ASSERT_EQ(out->dims()[1], 1000);
  //
  //  int step = 50;
  for (int i = 0; i < 10; i++) {
    // std::cout << ":" << out->data<float>()[i] << std::endl;
  }
  //  for (int i = 0; i < results.size(); ++i) {
  //    for (int j = 0; j < results[i].size(); ++j) {
  //      EXPECT_NEAR(out->data<float>()[j * step + (out->dims()[1] * i)],
  //                  results[i][j],
  //                  1e-6);
  //    }
  //  }

  std::string file = "plate_data/" + FLAGS_input_file.substr(9);
  std::cout << "file:::" << file << std::endl;

  std::ofstream ofs;
  ofs.open(file);
  for (int i = 0; i < out->dims().production(); i++) {
    float value = out->data<float>()[i];
    ofs << value << std::endl;
  }
  ofs.close();
}

TEST(OcrAttention, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  // Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},

  TestModel(valid_places, Place{TARGET(kARM), PRECISION(kFloat)});
}

}  // namespace lite
}  // namespace paddle
