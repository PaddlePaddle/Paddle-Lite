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

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "runtime/context.h"
#include "runtime/model.h"

namespace nnadapter {
namespace runtime {

class Compilation {
 public:
  class Buffer {
   public:
    Buffer() {}
    ~Buffer() {
      if (data) {
        free(data);
      }
    }

   public:
    void* data{nullptr};
    uint32_t size{0};
    std::vector<int32_t> dimensions;
  };
  class Program {
   public:
    Program() {}
    ~Program();

   public:
    Context::DeviceContext* device_context{nullptr};
    core::Cache* cache{nullptr};
    void* program{nullptr};
    // The following is the necessary information for the submodel from model
    // partition
    core::Model* model{nullptr};
    // Indicates where the model came from, whether it was externally
    // referenced, or created after model partition.
    bool referenced{true};
    // The relationship between the input and output operands of the submodels:
    // Negative value represents the input index of the entire model, otherwise
    // represents the index of operand shared between the submodels.
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;
  };
  Compilation(Model* model,
              const char* cache_token,
              void* cache_buffer,
              uint32_t cache_length,
              const char* cache_dir,
              Context* context);
  ~Compilation() {}
  int Finish();
  int QueryInputsAndOutputs(uint32_t* input_count,
                            NNAdapterOperandType** input_types,
                            uint32_t* output_count,
                            NNAdapterOperandType** output_types);
  int Execute(std::vector<core::Argument>* input_arguments,
              std::vector<core::Argument>* output_arguments);

 private:
  bool CheckCache();
  void ClearCache();
  int PartitionModel(
      Context* context,
      Model* model,
      std::vector<std::pair<
          Context::DeviceContext*,
          std::tuple<core::Model*, bool, std::vector<int>, std::vector<int>>>>*
          models);
  // Serialize/deserialize the cached models into/from memory
  bool Serialize(std::vector<uint8_t>* buffer);
  bool Deserialize(void* buffer, uint64_t size);

 private:
  Model* model_{nullptr};
  std::string cache_token_;
  std::string cache_dir_;
  std::vector<Program> programs_;
  std::vector<std::shared_ptr<Buffer>>
      buffers_;  // The shared buffers among the submodels
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  Context* context_{nullptr};
  bool completed_{false};
};

}  // namespace runtime
}  // namespace nnadapter
