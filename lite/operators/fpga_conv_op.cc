// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef LITE_WITH_FPGA
#include "lite/operators/fpga_conv_op.h"
#include <algorithm>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FpgaConvOpLite::InferShapeImpl() const {
  int fuse_idx = stride_info_.fuse_idx_;
  if (fuse_idx == 0) {
    ConvOpLite::InferShapeImpl();
    // additional config for fpga conv
    auto origin_shape = static_cast<ConvParam*>(op_param_)->output->dims();
    auto new_shape = origin_shape;
    new_shape[1] = stride_info_.wd_offset_;
    auto new_dim = DDimLite(new_shape);
    static_cast<ConvParam*>(op_param_)->output->Resize(new_dim);
  }

  // TODO how to guarantee the same input lod
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fpga_conv2d, paddle::lite::operators::FpgaConvOpLite);
#endif
