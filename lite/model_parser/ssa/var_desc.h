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

#include <memory>
#include <string>
#include <vector>
#include "lite/model_parser/general/var_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class OpDesc;

class VarDesc : public std::enable_shared_from_this<VarDesc> {
 public:
  VarDesc() = delete;

  VarDesc(const VarDesc&) = default;

  VarDesc(int32_t block_idx, const general::VarDesc* raw_desc)
      : block_idx_{block_idx}, meta_{std::make_shared<MetaVar>()} {
    meta_->raw_desc = raw_desc;
  }

  int32_t block_idx() const { return block_idx_; }

  void ResetBlockIdx(int32_t idx) { block_idx_ = idx; }

  const general::VarDesc* root_var_desc() const { return meta_->raw_desc; }

  std::string root_name() const { return meta_->raw_desc->Name(); }

  std::string mangled_name() const {
    std::string suffix;
    if (count_) {
      suffix = "__Mangled_" + std::to_string(count_);
    }
    return root_name() + suffix;
  }

  VarDataType GetType() const { return meta_->raw_desc->GetType(); }

  bool Persistable() const { return meta_->raw_desc->Persistable(); }

  std::weak_ptr<VarDesc> Read(const ssa::OpDesc& op_desc) {
    depends_.push_back(&op_desc);
    std::shared_ptr<VarDesc> desc;
    if (meta_->kids.empty()) {
      desc = shared_from_this();
    } else {
      desc = meta_->kids.back();
    }
    return desc;
  }

  std::weak_ptr<VarDesc> Written(const ssa::OpDesc& op_desc) {
    std::shared_ptr<VarDesc> desc;
    if (GetType() == VarDataType::LOD_TENSOR) {
      if (mutable_) {
        mutable_ = !mutable_;
        desc = shared_from_this();
      } else {
        meta_->kids.emplace_back(std::make_shared<VarDesc>(*this));
        desc = meta_->kids.back();
        desc->SetCount(meta_->kids.size());
      }
    } else {
      desc = shared_from_this();
    }
    return desc;
  }

  std::vector<std::weak_ptr<VarDesc>> kids() const {
    std::vector<std::weak_ptr<VarDesc>> kids;
    for (auto& kid : meta_->kids) {
      kids.emplace_back(kid);
    }
    return kids;
  }

 protected:
  void SetCount(size_t count) { count_ = count; }

 private:
  struct MetaVar {
    general::VarDesc const* raw_desc{nullptr};
    std::vector<std::shared_ptr<VarDesc>> kids;
  };
  int32_t block_idx_{-1};
  std::shared_ptr<MetaVar> meta_;
  bool mutable_{true};
  std::vector<OpDesc const*> depends_;
  size_t count_{0};
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
