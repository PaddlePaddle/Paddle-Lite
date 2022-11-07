// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/sequence_softmax_compute.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SequenceSoftmaxCompute::PrepareForRun() {
  lod_cpu_.reset(new int[XPU_MAX_LOD_SIZE]);
}

void SequenceSoftmaxCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto* in = param.X;
  auto* out = param.Out;
  // get lod
  auto seq_offset = in->lod()[0];
  for (size_t i = 0; i < seq_offset.size(); ++i) {
    lod_cpu_[i] = seq_offset[i];
  }
  // get shape
  auto input_dims = in->dims();
  std::vector<int> xshape;
  for (size_t i = 0; i < input_dims.size(); i++) {
    xshape.push_back(input_dims[i]);
  }
  int seq_num = seq_offset.size();
  int r = 0;
  r = xdnn::sequence_softmax<float>(ctx.GetRawContext(),
                                    in->data<float>(),
                                    out->mutable_data<float>(TARGET(kXPU)),
                                    xshape,
                                    0,
                                    {lod_cpu_.get(), seq_num, nullptr});
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_softmax,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SequenceSoftmaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
