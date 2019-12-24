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
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 1, 100, 200})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  read_from_file(FLAGS_input_file, data, 100 * 200);
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

  for (int i = 0; i < position_encoding->dims().production(); ++i) {
    temp_data[i] = 0;
  }
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
  // chw_to_hwc(temp_data, position_encoding_data, 33, 10, 23);
  // delete[] temp_data;

  // read_from_file("position_encoding.data", position_encoding_data, 33 * 10 *
  // 23);
  auto start = GetCurrentUS();
  for (int i = 0; i < 2; ++i) {
    predictor.Run();
  }

  std::cout << "================== Speed Report ===================";
  std::cout << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  auto* out = predictor.GetOutput(0);

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
  TestModel(valid_places, Place{TARGET(kARM), PRECISION(kFloat)});
}

}  // namespace lite
}  // namespace paddle
