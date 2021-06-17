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
#include <set>
#include <string>
#include <vector>

#include "lite/model_parser/general/op_desc.h"
#include "lite/model_parser/ssa/var_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class BlockDesc;

class OpDescBase {
 public:
  OpDescBase() = default;

  OpDescBase(const OpDescBase&) = default;

  virtual ~OpDescBase() = default;

  explicit OpDescBase(const general::OpDesc& raw_desc) : raw_desc_{&raw_desc} {}

  virtual const general::OpDesc& src_raw_desc() const {
    CHECK(raw_desc_);
    return *raw_desc_;
  }

  virtual std::string type() const { return raw_desc_->Type(); }

  const std::map<std::string, std::vector<std::weak_ptr<VarDesc>>>& inputs()
      const {
    return inputs_;
  }

  const std::map<std::string, std::vector<std::weak_ptr<VarDesc>>>& outputs()
      const {
    return outputs_;
  }

 protected:
  void UpdateVarBlockIdx(const std::weak_ptr<VarDesc>& var_desc,
                         int32_t op_block_idx);

  general::OpDesc const* raw_desc_{nullptr};
  std::map<std::string, std::vector<std::weak_ptr<VarDesc>>> inputs_;
  std::map<std::string, std::vector<std::weak_ptr<VarDesc>>> outputs_;
};

std::set<std::weak_ptr<VarDesc>, VarDescLT> ConvertToSet(
    const std::map<std::string, std::vector<std::weak_ptr<VarDesc>>>& map);

class OpDesc : public OpDescBase {
 public:
  OpDesc() = default;
  OpDesc(const general::OpDesc& raw_desc,
         const RootVarScope& scope,
         int32_t block_idx);

 private:
  std::weak_ptr<VarDesc> AddInput(const std::string& param,
                                  const std::weak_ptr<VarDesc>& desc);
  std::weak_ptr<VarDesc> AddOutput(const std::string& param,
                                   const std::weak_ptr<VarDesc>& desc);
};

class BlockParamInfo {
 public:
  static BlockParamInfo& instance();
  bool IsBlockOp(const std::string& op_type);
  const std::string& Attr(const std::string& op_type);
  const std::string& In(const std::string& op_type);
  const std::string& Out(const std::string& op_type);

 private:
  BlockParamInfo();
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
              const RootVarScope& scope,
              int32_t block_idx);

  void AddBlockInput(const std::weak_ptr<VarDesc>& var) {
    inputs_[block_in_param_].push_back(var);
  }
  void AddBlockOutput(const std::weak_ptr<VarDesc>& var) {
    outputs_[block_out_param_].push_back(var);
  }

 private:
  std::string block_in_param_;
  std::string block_out_param_;
};

class WriteBackOp : public OpDescBase {
 public:
  WriteBackOp(const std::weak_ptr<VarDesc>& src,
              const std::weak_ptr<VarDesc>& dst,
              int32_t block_idx);

  const general::OpDesc& src_raw_desc() const override { return fake_desc_; }

  std::string type() const override { return type_; }

 private:
  void AddInput(const std::string& param,
                const std::weak_ptr<VarDesc>& desc,
                int32_t block_idx);

  static constexpr const char* type_{"__WriteBack__"};
  static constexpr const char* input_deps_{"Dependencies"};
  static constexpr const char* input_src_{"X"};
  static constexpr const char* input_dst_{"Y"};
  general::OpDesc fake_desc_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
