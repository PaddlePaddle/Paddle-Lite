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

#ifndef TOOLS_QUANTIFICATION_SRC_BLOCK_DESC_LOCAL_H_
#define TOOLS_QUANTIFICATION_SRC_BLOCK_DESC_LOCAL_H_

#include <memory>
#include <vector>
#include "src/var_desc.h"

class BlockDesc {
 public:
  friend class Node;
  friend class ProgramOptimize;
  BlockDesc() {}
  explicit BlockDesc(PaddleMobile__Framework__Proto__BlockDesc *desc);

  const int &ID() const { return index_; }

  const bool &MultiThread() const { return multi_thread_; }

  const int &Parent() const { return parent_index_; }

  bool operator==(const BlockDesc &in_block) const {
    return this->ID() == in_block.ID() && this->Parent() == in_block.Parent();
  }

  bool operator<(const BlockDesc &in_block) const {
    return this->ID() < in_block.ID() && this->Parent() < in_block.Parent();
  }

  std::vector<std::shared_ptr<paddle_mobile::framework::VarDesc>> Vars() const;

 private:
  int index_;
  bool multi_thread_;
  int parent_index_;
  std::vector<std::shared_ptr<paddle_mobile::framework::VarDesc>> vars_;
};

#endif  // TOOLS_QUANTIFICATION_SRC_BLOCK_DESC_LOCAL_H_
