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

#include "lite/core/device_info.h"

namespace paddle {
namespace lite {

template <TargetType Type>
class Context;
using ContextMetal = Context<TargetType::kMetal>;

template <>
class Context<TargetType::kMetal> {
 public:
  void InitOnce();
  void CopySharedTo(ContextMetal* ctx);

  void* context() { return context_; }

 private:
  void* context_;
};

}  // namespace lite
}  // namespace paddle
