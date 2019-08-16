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
#include <algorithm>

namespace paddle_mobile {
namespace framework {

std::vector<std::shared_ptr<VarDesc>> BlockDesc::Vars() const { return vars_; }

std::vector<std::shared_ptr<OpDesc>> BlockDesc::Ops() const { return ops_; }

BlockDesc::BlockDesc(PaddleMobile__Framework__Proto__BlockDesc *desc)
    : index_(desc->idx), parent_index_(desc->idx) {
  for (int i = 0; i < desc->n_vars; ++i) {
    PaddleMobile__Framework__Proto__VarDesc *var_desc = desc->vars[i];
    vars_.emplace_back(std::shared_ptr<VarDesc>(new VarDesc(var_desc)));
  }

  std::sort(vars_.begin(), vars_.end(),
            [](std::shared_ptr<VarDesc> left, std::shared_ptr<VarDesc> right) {
              return left->Name() < right->Name();
            });

  for (int j = 0; j < desc->n_ops; ++j) {
    PaddleMobile__Framework__Proto__OpDesc *op_desc = desc->ops[j];
    ops_.emplace_back(new framework::OpDesc(op_desc));
  }
}

}  // namespace framework
}  // namespace paddle_mobile
