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
// Created by 谢柏渊 on 2018/7/25.
//

#include "src/program_desc.h"
#include <vector>

ProgramDesc::ProgramDesc(PaddleMobile__Framework__Proto__ProgramDesc *desc) {
  for (int i = 0; i < desc->n_blocks; ++i) {
    blocks_.emplace_back(std::make_shared<BlockDesc>(desc->blocks[i]));
  }
}

const std::vector<std::shared_ptr<BlockDesc>> ProgramDesc::Blocks() {
  return blocks_;
}
