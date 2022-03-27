// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/nvidia_tensorrt/calibrator.h"
#include <functional>
#include <map>
#include <numeric>
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

Int8EntropyCalibrator::Int8EntropyCalibrator(int batch_size,
                                             std::string dataset_path,
                                             std::string table_path)
    : batch_size_(batch_size),
      dataset_path_(dataset_path),
      table_path_(table_path),
      index_(0) {
  if (!dataset_path.empty()) {
    // Parser dataset file names
    std::string dataset_file = dataset_path + "/lists.txt";
    auto lists = ReadLines(dataset_file);
    NNADAPTER_CHECK(!lists.empty()) << "Dataset list file:" << dataset_file
                                    << " has no data.";
    if (lists.size() < static_cast<size_t>(batch_size)) {
      NNADAPTER_LOG(WARNING) << "Input file is not enough. Only receive "
                             << lists.size() << " batch input.";
    }
    lists.resize(static_cast<int>(lists.size() / batch_size) * batch_size);
    input_file_names_.resize(string_split(lists.front(), ";").size());
    for (size_t i = 0; i < lists.size(); i++) {
      auto input_info = string_split(lists.at(i), ";");
      NNADAPTER_CHECK_EQ(input_info.size(), input_file_names_.size());
      for (size_t j = 0; j < input_info.size(); j++) {
        auto file_name = string_split(input_info.at(j), ":").back();
        input_file_names_.at(j).push_back(file_name);
      }
    }
    // Malloc input buffer
    auto input_info = string_split(lists.front(), ";");
    for (size_t i = 0; i < input_info.size(); i++) {
      auto shape =
          string_split<int>(string_split(input_info.at(i), ":").at(0), ",");
      auto precision = string_split(input_info.at(i), ":").at(1);
      std::map<std::string, int> precision_length_map{
          {"float32", 4}, {"float16", 2}, {"int8", 1}};
      int size = std::accumulate(
                     shape.begin(), shape.end(), 1, std::multiplies<int>()) *
                 precision_length_map.at(precision);
      void* data_ptr{nullptr};
      NNADAPTER_CHECK_EQ(cudaMalloc(&data_ptr, size), cudaSuccess);
      std::shared_ptr<void> device_buffer(data_ptr, [](void* ptr) {
        NNADAPTER_CHECK_EQ(cudaFree(ptr), cudaSuccess);
      });
      device_buffers_.push_back(device_buffer);
    }
  }
}

bool Int8EntropyCalibrator::getBatch(void* bindings[],
                                     const char* names[],
                                     int nbBindings) {
  // TODO(zhupengyang): support multi inputs
  NNADAPTER_CHECK_EQ(nbBindings, 1);
  if (static_cast<size_t>(index_) >= input_file_names_.at(0).size()) {
    return false;
  }
  std::vector<uint8_t> host_buffer;
  for (int i = 0; i < batch_size_; i++) {
    std::vector<uint8_t> host_data;
    std::string file_path =
        dataset_path_ + "/" + input_file_names_.at(0).at(index_ + i);
    NNADAPTER_CHECK(ReadFile(file_path, &host_data));
    host_buffer.insert(host_buffer.end(), host_data.begin(), host_data.end());
  }
  void* device_buffer = device_buffers_.at(0).get();
  NNADAPTER_CHECK_EQ(cudaMemcpy(device_buffer,
                                host_buffer.data(),
                                host_buffer.size(),
                                cudaMemcpyHostToDevice),
                     cudaSuccess);
  bindings[0] = device_buffer;
  index_ += batch_size_;
  return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) {
  if (table_path_.empty()) {
    NNADAPTER_LOG(WARNING) << "No calibration table file is set. New "
                              "calibration table will be generated.";
  }
  if (!ReadFile(table_path_, &table_)) {
    NNADAPTER_LOG(WARNING) << "Read calibration table file failed. New "
                              "calibration table will be generated.";
    return nullptr;
  }
  NNADAPTER_LOG(INFO) << "Read calibration table file from " << table_path_
                      << " success.";
  length = table_.size();
  return table_.data();
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache,
                                                  size_t length) {
  if (table_path_.empty()) {
    NNADAPTER_LOG(WARNING) << "No calibration table will be saved because "
                              "table_path is not found.";
    return;
  }
  auto data = reinterpret_cast<const uint8_t*>(cache);
  table_ = std::vector<uint8_t>(data, data + length);
  NNADAPTER_CHECK(WriteFile(table_path_, table_));
  NNADAPTER_LOG(INFO) << "Write calibration table to " << table_path_
                      << " success.";
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
