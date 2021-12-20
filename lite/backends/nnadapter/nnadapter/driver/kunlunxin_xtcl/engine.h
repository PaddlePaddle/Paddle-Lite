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
#include <vector>
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "driver/kunlunxin_xtcl/utility.h"
#include "utility/string.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Build(hal::Model* model, hal::Cache* cache);
  int Execute(uint32_t input_count,
              hal::Argument* input_arguments,
              uint32_t output_count,
              hal::Argument* output_arguments);

 private:
  void Clear();

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to XTCL expr
  std::map<hal::Operand*, std::vector<xtcl::xExpr>> exprs_;
  xtcl::network::xTensorCompiler::ParamNDArrayMap params_;
  xtcl::network::xNetworkBuilder builder_;
  std::unique_ptr<xtcl::network::xRuntimeInstance> runtime_{nullptr};
  std::vector<DLTensor> input_tensors_{};
  std::vector<DLTensor> output_tensors_{};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
};

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
