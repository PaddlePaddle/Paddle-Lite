// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "adnn_types.h"  // NOLINT

namespace adnn {

Status initialize(const struct Device* device) { return SUCCESS; }

void finalize() {}

Context* create_context() { return new Context(); }

void destroy_context(Context* context) {
  if (context) {
    delete context;
  }
}

}  // namespace adnn
