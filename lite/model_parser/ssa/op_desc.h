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
#include <utility>
#include <vector>

#include "lite/model_parser/general/op_desc.h"
#include "lite/model_parser/ssa/op_proto.h"
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

// In order to modify the block operator, we need to know the specific
// input name. Because its format is not uniform, so register here.

class BlockOpDesc : public OpDescBase {
 public:
  BlockOpDesc(const general::OpDesc& raw_desc,
              const std::shared_ptr<BlockOpProto>& proto)
      : OpDescBase{raw_desc}, proto_{proto} {}

  void AddBlockInput(const std::weak_ptr<VarDesc>& var) {
    inputs_[proto_->InKey()].push_back(var);
  }
  void AddBlockOutput(const std::weak_ptr<VarDesc>& var) {
    outputs_[proto_->OutKey()].push_back(var);
  }
  const std::vector<std::weak_ptr<VarDesc>>& extra_inputs() const {
    return extra_inputs_;
  }
  std::weak_ptr<BlockOpProto> proto() const { return proto_; }

 protected:
  std::shared_ptr<BlockOpProto> proto_;
  std::vector<std::weak_ptr<VarDesc>> extra_inputs_;
};

class WhileOp : public BlockOpDesc {
 public:
  WhileOp(const general::OpDesc& raw_desc,
          const RootVarScope& scope,
          int32_t block_idx)
      : BlockOpDesc{
            raw_desc,
            BlockOpProtoRegistry::instance().GetProto(raw_desc.Type())} {
    for (const auto& var : raw_desc.Input(cond_key_)) {
      auto var_desc = scope.GetRootVarDesc(var).lock()->Read(*this);
      inputs_[cond_key_].emplace_back(var_desc);
      extra_inputs_.emplace_back(var_desc);
    }
  }

 private:
  // The condition variable is the key implicit input reference
  // of the while operator.
  const std::string cond_key_{"Condition"};
};

class FakeBlockOp : public BlockOpDesc {
 public:
  FakeBlockOp(const general::OpDesc& raw_desc,
              const RootVarScope& scope,
              int32_t block_idx)
      : BlockOpDesc{
            raw_desc,
            BlockOpProtoRegistry::instance().GetProto(raw_desc.Type())} {}
};

class ConditionalBlockOp : public BlockOpDesc {
 public:
  ConditionalBlockOp(const general::OpDesc& raw_desc,
                     const RootVarScope& scope,
                     int32_t block_idx)
      : BlockOpDesc{
            raw_desc,
            BlockOpProtoRegistry::instance().GetProto(raw_desc.Type())} {}
};

class BlockOpGen {
 public:
  BlockOpGen() {
    Register("while",
             [](const general::OpDesc& raw_desc,
                const RootVarScope& scope,
                int32_t block_idx) {
               return std::unique_ptr<BlockOpDesc>(
                   new WhileOp(raw_desc, scope, block_idx));
             });
    Register("fake_block_op",
             [](const general::OpDesc& raw_desc,
                const RootVarScope& scope,
                int32_t block_idx) {
               return std::unique_ptr<BlockOpDesc>(
                   new FakeBlockOp(raw_desc, scope, block_idx));
             });
    Register("conditional_block",
             [](const general::OpDesc& raw_desc,
                const RootVarScope& scope,
                int32_t block_idx) {
               return std::unique_ptr<BlockOpDesc>(
                   new ConditionalBlockOp(raw_desc, scope, block_idx));
             });
  }

  bool IsBlockOp(const std::string& op_type) {
    return ctors_.find(op_type) != ctors_.end();
  }

  static BlockOpGen& instance() {
    static BlockOpGen instance_;
    return instance_;
  }

  std::unique_ptr<BlockOpDesc> NewOp(const general::OpDesc& raw_desc,
                                     const RootVarScope& scope,
                                     int32_t block_idx) {
    return ctors_.at(raw_desc.Type())(raw_desc, scope, block_idx);
  }

 private:
  using func_t = std::function<std::unique_ptr<BlockOpDesc>(
      const general::OpDesc&, const RootVarScope&, int32_t)>;

 private:
  BlockOpGen& Register(const std::string& op_type, func_t&& ctor) {
    ctors_[op_type] = std::move(ctor);
    return *this;
  }
  std::map<std::string, func_t> ctors_;
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

  static constexpr char type_[]{"write_back"};
  // In order to adapt to the operator registration system,
  // the dependent parameters are classified by variable types here.
  static constexpr char input_lod_deps_[]{"Dep_LoDTensor"};
  static constexpr char input_lod_array_deps_[]{"Dep_LoDTensorArray"};
  static constexpr char input_src_[]{"Src_LoDTensor"};
  // For directed acyclic, input is used as output here.
  static constexpr char input_dst_[]{"Dst_LoDTensor"};
  general::OpDesc fake_desc_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
