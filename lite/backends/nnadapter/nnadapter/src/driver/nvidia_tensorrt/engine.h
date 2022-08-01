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

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "driver/nvidia_tensorrt/program.h"
#include "driver/nvidia_tensorrt/utility.h"
#include "optimizer/partition_model_into_submodels.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program() {}

  int Build(core::Model* model, core::Cache* cache);
  int Execute(uint32_t input_count,
              core::Argument* input_arguments,
              uint32_t output_count,
              core::Argument* output_arguments);

 private:
  void Clear();
  int BuildFromModel(core::Model* model);
  int BuildFromCache(core::Cache* cache);
  int CheckInputsAndOutputs(uint32_t input_count,
                            core::Argument* input_arguments,
                            uint32_t output_count,
                            core::Argument* output_arguments);
  int SerializeToCache(std::vector<uint8_t>* buffer);
  int DeserializeFromCache(std::vector<uint8_t>* buffer);

 private:
  Context* context_;
  std::vector<std::pair<
      int,
      std::tuple<core::Model*, bool, std::vector<int>, std::vector<int>>>>
      sub_models_;
  std::vector<std::vector<uint8_t>> sub_caches_;
  std::vector<std::shared_ptr<ProgramBase>> sub_programs_;
  std::map<int, std::shared_ptr<Tensor>> input_tensors_;
  std::map<int, std::shared_ptr<Tensor>> temporary_tensors_;
  std::map<int, std::shared_ptr<Tensor>> output_tensors_;
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  int max_batch_size_{1};
  bool is_sub_model_from_cache_{false};
  bool io_use_device_buffer_{false};
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
