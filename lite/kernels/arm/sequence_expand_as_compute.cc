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

#include "lite/kernels/arm/sequence_expand_as_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void SequenceExpandAsCompute::Run() {
  auto& param = Param<operators::SequenceExpandAsParam>();
  auto* x = param.x;
  auto* y = param.y;
  auto* out = param.out;
  auto y_lod = y->lod();
  CHECK_EQ(y_lod.size(), 1u);
  CHECK_GT(y_lod[0].size(), 1u);

  auto dims = x->dims();
  auto out_data = out->mutable_data<float>();
  auto x_data = x->data<float>();
  int seq_size = x->numel() / dims[0];

  std::vector<uint64_t> out_lod;
  out_lod.push_back(0);
  int sum = 0;
  for (int i = 1; i < y_lod[0].size(); i++) {
    int repeat_num = y_lod[0][i] - y_lod[0][i - 1];
    if (repeat_num == 0) {
      continue;
    } else {
      for (int j = 0; j < repeat_num; j++) {
        memcpy(out_data, x_data, sizeof(float) * seq_size);
        out_data += seq_size;
      }
      x_data += seq_size;
    }
    sum += repeat_num;
    out_lod.push_back(sum);
  }
  std::vector<std::vector<uint64_t>> lod;
  lod.push_back(out_lod);
  out->set_lod(lod);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_expand_as,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SequenceExpandAsCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
