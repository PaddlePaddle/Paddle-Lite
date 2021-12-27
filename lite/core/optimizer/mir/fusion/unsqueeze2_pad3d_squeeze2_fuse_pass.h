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
//
// Created by chenyaohuang on 2021/12/17.
//

// unsqueeze2->pad3d->squeeze2 =>change to pad2d;
#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {
// This pass fuse unsqueeze2 pad3d and squeeze2.
// unsqueeze2 ->pad3d -> squeeze2 == pad2d
// unsqueeze2 and squeeze2 are essentially the dimension increase and dimension
// reduction of input data.
// Therefore, this pass can reduce the operation of the kernel. And pad2d
// supports more kinds (OpenCL, arm..)

class Unsqueeze2Pad3dSqueeze2FusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
