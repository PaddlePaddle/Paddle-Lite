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

#include "lite/operators/gaussian_random_op.h"
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool GaussRandomOp::CheckShape() const {
  auto* X = param_.X;
  auto* OutSize = param_.OutSize;
  CHECK_OR_FALSE(X);
  if (OutSize != nullptr) {
    CHECK_OR_FALSE(OutSize);
  }
  CHECK_OR_FALSE(param_.Out);
  return true;
}


} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(gaussian_random, paddle::lite::operators::GaussRandomOp);
