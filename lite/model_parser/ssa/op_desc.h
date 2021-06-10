// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/model_parser/general/op_desc.h"
#include "lite/model_parser/ssa/scope.h"
#include "lite/model_parser/ssa/var_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class BlockDesc;

class OpDescBase {
 public:
  OpDescBase() = delete;

  OpDescBase(const OpDescBase&) = default;

  explicit OpDescBase(const general::OpDesc& raw_desc) : raw_desc_{&raw_desc} {}

  const general::OpDesc& src_raw_desc() const {
    CHECK(raw_desc_);
    return *raw_desc_;
  }

  std::string type() const { return raw_desc_->Type(); }

  const std::map<std::string, std::vector<std::weak_ptr<const ssa::VarDesc>>>&
  inputs() const {
    return inputs_;
  }

  const std::map<std::string, std::vector<std::weak_ptr<const ssa::VarDesc>>>&
  outputs() const {
    return outputs_;
  }

 protected:
  general::OpDesc const* raw_desc_{nullptr};
  std::map<std::string, std::vector<std::weak_ptr<const ssa::VarDesc>>> inputs_;
  std::map<std::string, std::vector<std::weak_ptr<const ssa::VarDesc>>>
      outputs_;
};

class OpDesc : public OpDescBase {
 public:
  OpDesc(const general::OpDesc& raw_desc, const Scope& scope, int32_t block_idx)
      : OpDescBase{raw_desc} {
    for (const auto& param : raw_desc.InputArgumentNames()) {
      for (const auto& var : raw_desc.inputs().at(param)) {
        const auto& var_desc = AddInput(param, scope.GetRootVarDesc(var));
        UpdateVarBlockIdx(var_desc, block_idx);
      }
    }
    for (const auto& param : raw_desc.OutputArgumentNames()) {
      for (const auto& var : raw_desc.outputs().at(param)) {
        const auto& var_desc = AddOutput(param, scope.GetRootVarDesc(var));
        UpdateVarBlockIdx(var_desc, block_idx);
      }
    }
  }

 private:
  void UpdateVarBlockIdx(const std::weak_ptr<VarDesc>& var_desc,
                         int32_t op_block_idx) {
    if (op_block_idx < var_desc.lock()->block_idx()) {
      var_desc.lock()->ResetBlockIdx(op_block_idx);
    }
  }

  std::weak_ptr<ssa::VarDesc> AddInput(
      const std::string& param, const std::weak_ptr<ssa::VarDesc>& desc) {
    auto var_desc{desc.lock()->Read(*this)};
    inputs_[param].emplace_back(var_desc);
    return var_desc;
  }

  std::weak_ptr<ssa::VarDesc> AddOutput(
      const std::string& param, const std::weak_ptr<ssa::VarDesc>& desc) {
    auto var_desc{desc.lock()->Written(*this)};
    outputs_[param].emplace_back(var_desc);
    return var_desc;
  }
};

class BlockParamInfo {
 public:
  static BlockParamInfo& instance() {
    static BlockParamInfo instance_;
    return instance_;
  }
  bool IsBlockOp(const std::string& op_type) {
    return op_block_param_.find(op_type) != op_block_param_.end();
  }
  const std::string& GetBlockAttrName(const std::string& op_type) {
    return op_block_param_.at(op_type).attr;
  }
  const std::string& GetBlockInputName(const std::string& op_type) {
    return op_block_param_.at(op_type).input;
  }
  const std::string& GetBlockOutputName(const std::string& op_type) {
    return op_block_param_.at(op_type).output;
  }

 private:
  BlockParamInfo() {
    op_block_param_ = {
        {"while", {"block_idx", "kX", "kOutput"}},
    };
  }
  struct Param {
    std::string attr;
    std::string input;
    std::string output;
  };
  std::map<std::string, Param> op_block_param_;
};

class BlockOpDesc : public OpDescBase {
 public:
  BlockOpDesc(const general::OpDesc& raw_desc,
              const Scope& scope,
              int32_t block_idx)
      : OpDescBase{raw_desc} {
    CHECK(BlockParamInfo::instance().IsBlockOp(raw_desc.Type()));
    for (const auto& param : raw_desc.InputArgumentNames()) {
      for (const auto& var : raw_desc.inputs().at(param)) {
        inputs_[param].emplace_back(scope.GetRootVarDesc(var));
      }
    }
    for (const auto& param : raw_desc.OutputArgumentNames()) {
      for (const auto& var : raw_desc.outputs().at(param)) {
        outputs_[param].emplace_back(scope.GetRootVarDesc(var));
      }
    }
  }

  void ResetBlockInput(std::vector<std::weak_ptr<const ssa::VarDesc>>&& vars) {
    const std::string& param{
        BlockParamInfo::instance().GetBlockInputName(raw_desc_->Type())};
    std::swap(inputs_[param], vars);
  }
  void ResetBlockOutput(std::vector<std::weak_ptr<const ssa::VarDesc>>&& vars) {
    const std::string& param{
        BlockParamInfo::instance().GetBlockOutputName(raw_desc_->Type())};
    std::swap(outputs_[param], vars);
  }
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
