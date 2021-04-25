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
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

class Operand {
 public:
  enum class Lifetime {
    kTemporaryVariable = 0,
    kConstant = 1,
    kInput = 2,
    kOutput = 3
  };

  Operand(NNAdapterOperand* operand, Lifetime lifetime)
      : operand_(operand), lifetime_(lifetime) {}

  void set_operand(NNAdapterOperand* operand) { operand_ = operand; }
  void set_lifetime(Lifetime lifetime) { lifetime_ = lifetime; }

  NNAdapterOperand* operand() { return operand_; }
  bool is_temporary_variable() const {
    return lifetime_ == Lifetime::kTemporaryVariable;
  }
  bool is_constant() const { return lifetime_ == Lifetime::kConstant; }
  bool is_input() const { return lifetime_ == Lifetime::kInput; }
  bool is_output() const { return lifetime_ == Lifetime::kOutput; }

 private:
  NNAdapterOperand* operand_{nullptr};
  Lifetime lifetime_{Lifetime::kTemporaryVariable};
};

class Converter {
 public:
  explicit Converter(const std::vector<std::string>& device_names) {
    for (auto& device_name : device_names) {
      NNAdapterDevice* device = nullptr;
      int result = NNAdapterDevice_acquire(device_name.c_str(), &device);
      bool found = result == NNADAPTER_NO_ERROR && device != nullptr;
      if (found) {
        const char* name = nullptr;
        NNAdapterDevice_getName(device, &name);
        const char* vendor = nullptr;
        NNAdapterDevice_getVendor(device, &vendor);
        NNAdapterDeviceType type = 0;
        NNAdapterDevice_getType(device, &type);
        int32_t version = 0;
        NNAdapterDevice_getVersion(device, &version);
        LOG(INFO) << device_name << "(" << name << ":" << vendor << ":" << type
                  << ":" << version << ")";
        devices_.push_back(device);
      }
    }
    NNAdapterModel_create(&model_);
  }

  ~Converter() {
    NNAdapterModel_destroy(model_);
    for (auto* device : devices_) {
      NNAdapterDevice_release(device);
    }
  }

 public:
  NNAdapterModel* GetModel() { return model_; }

  int AddOperand(const std::string& name, std::shared_ptr<Operand> operand);
  std::shared_ptr<Operand> AddOperand(const std::string& name,
                                      NNAdapterOperand* operand);

  std::shared_ptr<Operand> GetOperand(std::string name) {
    CHECK(HasOperand(name)) << "Node " << name << " not found.";
    return operands_.at(name).back();
  }

  bool HasOperand(const std::string& name) {
    return operands_.find(name) != operands_.end();
  }

 private:
  std::map<std::string, std::vector<std::shared_ptr<Operand>>> operands_;
  NNAdapterModel* model_{nullptr};
  std::vector<NNAdapterDevice*> devices_;
};

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
