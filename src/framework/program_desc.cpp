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

//
// Created by liuRuiLong on 2018/5/4.
//

#include "framework/program_desc.h"

namespace paddle_mobile {
namespace framework {

ProgramDesc::ProgramDesc(const proto::ProgramDesc &desc) : desc_(desc) {
  for (auto &block_desc : *desc_.mutable_blocks()) {
    // new framework::BlockDesc(block_desc)
    blocks_.emplace_back(std::make_shared<BlockDesc>(block_desc));
  }
}

std::shared_ptr<BlockDesc> ProgramDesc::Block(size_t idx) {
  return blocks_[idx];
}

}  // namespace framework
}  // namespace paddle_mobile
