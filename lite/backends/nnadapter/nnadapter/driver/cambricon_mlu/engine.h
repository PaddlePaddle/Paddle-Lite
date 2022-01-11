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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "driver/cambricon_mlu/utility.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  std::string GetBuildConfigFilePath() { return build_config_file_path_; }
  ~Context();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  std::string build_config_file_path_ = "";
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {
    NNADAPTER_VLOG(3) << "Creating MagicMind IBuilder";
    mm_builder_.reset(magicmind::CreateIBuilder());
    NNADAPTER_VLOG(3) << "Creating MagicMind IBuilderConfig";
    mm_builder_config_.reset(magicmind::CreateIBuilderConfig());
    NNADAPTER_VLOG(3) << "Creating MagicMind INetwork";
    mm_network_.reset(magicmind::CreateINetwork());
    NNADAPTER_VLOG(3) << "Creating queue";
    cnrtCreateQueue(&queue_);
  }
  ~Program();

  int Build(hal::Model* model, hal::Cache* cache);
  int Execute(uint32_t input_count,
              hal::Argument* input_arguments,
              uint32_t output_count,
              hal::Argument* output_arguments);
  bool CheckShapeValid();

 private:
  void Clear();
  void SetValidShapes(const std::vector<NNAdapterOperandType>& input_types);
  // Build from model or cache
  int BuildFromModel(hal::Model* model);
  int BuildFromCache(hal::Cache* cache);

 private:
  Context* context_{nullptr};
  cnrtQueue_t queue_;
  MMUniquePtrType<magicmind::IBuilder> mm_builder_;
  MMUniquePtrType<magicmind::IBuilderConfig> mm_builder_config_;
  MMUniquePtrType<magicmind::INetwork> mm_network_;
  MMUniquePtrType<magicmind::IModel> mm_model_;
  MMUniquePtrType<magicmind::IEngine> mm_engine_;
  MMUniquePtrType<magicmind::IContext> mm_context_;
  // Map NNAdapter operand to magicmind itensor
  std::map<hal::Operand*, std::vector<magicmind::ITensor*>> tensors_;
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::vector<std::vector<int32_t>> valid_shapes_;
  std::vector<std::string> input_names_;
  std::string dump_graph_path_;
  std::vector<uint8_t>* dump_graph_buffer_{nullptr};
};

}  // namespace cambricon_mlu
}  // namespace nnadapter
