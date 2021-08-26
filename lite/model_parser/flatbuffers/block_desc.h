// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "lite/core/model/base/block_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/model_parser/flatbuffers/op_desc.h"
#include "lite/model_parser/flatbuffers/var_desc.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class BlockDescView : public BlockDescAPI {
 public:
  BlockDescView() = default;

  BlockDescView(const BlockDescView&) = delete;

  explicit BlockDescView(proto::BlockDesc const* desc) : desc_(desc) {
    CHECK(desc_);
    vars_.resize(VarsSize());
    ops_.resize(OpsSize());
    for (size_t idx = 0; idx < VarsSize(); ++idx) {
      vars_[idx].reset(new VarDescView(desc_->vars()->Get(idx)));
    }
    for (size_t idx = 0; idx < OpsSize(); ++idx) {
      ops_[idx].reset(new OpDescView(desc_->ops()->Get(idx)));
    }
  }

  int32_t Idx() const override { return desc_->idx(); }

  int32_t ParentIdx() const override { return desc_->parent_idx(); }

  size_t VarsSize() const override { return desc_->vars()->size(); }

  template <typename T>
  T const* GetVar(int32_t idx) const;

  template <typename T>
  T* GetVar(int32_t idx) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  size_t OpsSize() const override {
    CHECK(desc_);
    CHECK(desc_->ops());
    return desc_->ops()->size();
  }

  template <typename T>
  T const* GetOp(int32_t idx) const;

  template <typename T>
  T* GetOp(int32_t idx) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  const std::vector<std::unique_ptr<VarDescView>>& GetVars() const {
    return vars_;
  }

  int32_t ForwardBlockIdx() const override {
    return desc_->forward_block_idx();
  }

 private:
  proto::BlockDesc const* desc_;  // not_own
  std::vector<std::unique_ptr<VarDescView>> vars_;
  std::vector<std::unique_ptr<OpDescView>> ops_;
};

#ifdef LITE_WITH_FLATBUFFERS_DESC
class BlockDesc : public BlockDescAPI {
 public:
  BlockDesc() : owned_(true), desc_(new proto::BlockDescT()) {}

  BlockDesc(const BlockDesc&) = delete;

  explicit BlockDesc(proto::BlockDescT* desc) : desc_(desc) {
    CHECK(desc_);
    SyncVars();
    SyncOps();
  }

  int32_t Idx() const override { return desc_->idx; }

  void SetIdx(int32_t idx) override { desc_->idx = idx; }

  int32_t ParentIdx() const override { return desc_->parent_idx; }

  void SetParentIdx(int32_t idx) override { desc_->parent_idx = idx; }

  size_t VarsSize() const override { return desc_->vars.size(); }

  void ClearVars() override {
    desc_->vars.clear();
    SyncVars();
  }

  size_t OpsSize() const override { return desc_->ops.size(); }

  void ClearOps() override {
    desc_->ops.clear();
    SyncOps();
  }

  int32_t ForwardBlockIdx() const override { return desc_->forward_block_idx; }

  void SetForwardBlockIdx(int32_t idx_in) override {
    desc_->forward_block_idx = idx_in;
  }

  proto::BlockDescT* raw_desc() { return desc_; }

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  T* AddVar();

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  T* AddOp();

  ~BlockDesc() {
    if (owned_) {
      delete desc_;
    }
  }

 private:
  void SyncVars() {
    vars_.resize(desc_->vars.size());
    for (size_t i = 0; i < desc_->vars.size(); ++i) {
      if (!vars_[i] || vars_[i]->raw_desc() != desc_->vars[i].get()) {
        vars_[i].reset(new VarDesc(desc_->vars[i].get()));
      }
    }
  }
  void SyncOps() {
    ops_.resize(desc_->ops.size());
    for (size_t i = 0; i < desc_->ops.size(); ++i) {
      if (!ops_[i] || ops_[i]->raw_desc() != desc_->ops[i].get()) {
        ops_[i].reset(new OpDesc(desc_->ops[i].get()));
      }
    }
  }

  bool owned_{false};
  proto::BlockDescT* desc_{nullptr};
  std::vector<std::unique_ptr<VarDesc>> vars_;
  std::vector<std::unique_ptr<OpDesc>> ops_;
};
#endif  // LITE_WITH_FLATBUFFERS_DESC

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
