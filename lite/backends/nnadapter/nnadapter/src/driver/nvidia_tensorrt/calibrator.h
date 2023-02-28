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

#pragma once
#include <memory>
#include <string>
#include <vector>
#include "driver/nvidia_tensorrt/utility.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  Int8EntropyCalibrator(int batch_size,
                        std::string dataset_path,
                        std::string table_path);
  virtual ~Int8EntropyCalibrator() TRT_NOEXCEPT {}

  int getBatchSize() const TRT_NOEXCEPT override { return batch_size_; }
  bool getBatch(void* bindings[],
                const char* names[],
                int nb_bindings) TRT_NOEXCEPT override;
  const void* readCalibrationCache(size_t& length) TRT_NOEXCEPT override;
  void writeCalibrationCache(const void* cache,
                             size_t length) TRT_NOEXCEPT override;

 private:
  int batch_size_{1};
  std::string dataset_path_;
  std::string table_path_;
  int index_{0};
  std::vector<std::vector<std::string>> input_file_names_;
  std::vector<std::shared_ptr<void>> device_buffers_;
  std::vector<uint8_t> table_;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
