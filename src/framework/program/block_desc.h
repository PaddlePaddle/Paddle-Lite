/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <vector>
#include "framework/framework.pb-c.h"
#include "framework/program/op_desc.h"
#include "framework/program/var_desc.h"

namespace paddle_mobile {
namespace framework {

class BlockDesc {
 public:
  friend class Node;
  friend class ProgramOptimize;
  BlockDesc() {}
  explicit BlockDesc(PaddleMobile__Framework__Proto__BlockDesc *desc);
  explicit BlockDesc(const BlockDesc &block_desc)
      : index_(block_desc.index_), parent_index_(block_desc.parent_index_) {
    for (auto &op_desc : block_desc.ops_) {
      std::shared_ptr<OpDesc> copy_op_desc = std::make_shared<OpDesc>(*op_desc);
      ops_.push_back(copy_op_desc);
    }

    for (int i = 0; i < block_desc.vars_.size(); ++i) {
      auto &var_desc = block_desc.vars_[i];
      vars_.emplace_back(std::make_shared<VarDesc>(*var_desc));
    }
  }

  const int &ID() const { return index_; }

  const bool &MultiThread() const { return multi_thread_; }

  const int &Parent() const { return parent_index_; }

  bool operator==(const paddle_mobile::framework::BlockDesc &in_block) const {
    return this->ID() == in_block.ID() && this->Parent() == in_block.Parent();
  }

  bool operator<(const paddle_mobile::framework::BlockDesc &in_block) const {
    return this->ID() < in_block.ID() && this->Parent() < in_block.Parent();
  }

  std::vector<std::shared_ptr<VarDesc>> Vars() const;
  std::vector<std::shared_ptr<OpDesc>> Ops() const;

 private:
  int index_;
  bool multi_thread_;
  int parent_index_;
  std::vector<std::shared_ptr<OpDesc>> ops_;
  std::vector<std::shared_ptr<VarDesc>> vars_;
};

}  // namespace framework
}  // namespace paddle_mobile

namespace std {

template <>
struct hash<paddle_mobile::framework::BlockDesc> {
  typedef paddle_mobile::framework::BlockDesc argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type const &s) const noexcept {
    result_type const h1(std::hash<int>{}(s.ID()));
    result_type const h2(std::hash<int>{}(s.ID()));
    return h1 ^ (h2 << 1);
  }
};

}  // namespace std
