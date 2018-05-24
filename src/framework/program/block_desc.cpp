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

#include "block_desc.h"

namespace paddle_mobile {
namespace framework {

std::vector<std::shared_ptr<VarDesc>> BlockDesc::Vars() const {
  std::vector<std::shared_ptr<VarDesc>> res;
  for (const auto &p : vars_) {
    res.push_back(p.second);
  }
  return res;
}

std::vector<std::shared_ptr<OpDesc>> BlockDesc::Ops() const {
  std::vector<std::shared_ptr<OpDesc>> res;
  for (const auto &op : ops_) {
    res.push_back(op);
  }
  return res;
}

BlockDesc::BlockDesc(const proto::BlockDesc &desc):
        index_(desc.idx()), parent_index_(desc.parent_idx()) {
  for (const proto::VarDesc &var_desc : desc.vars()) {
    vars_[var_desc.name()].reset(new VarDesc(var_desc));
  }
  for (const proto::OpDesc &op_desc : desc.ops()) {
    ops_.emplace_back(new framework::OpDesc(op_desc));
  }
}

}  // namespace framework
}  // namespace paddle_mobile
