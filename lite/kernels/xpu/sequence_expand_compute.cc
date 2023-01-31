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

#include "lite/kernels/xpu/sequence_expand_compute.h"
#include <algorithm>
#include <memory>
#include <vector>

#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void SequenceExpandCompute<T, PType>::PrepareForRun() {
  lodx_cpu_.reset(new int[XPU_MAX_LOD_SIZE_64]);
  lody_cpu_.reset(new int[XPU_MAX_LOD_SIZE_64]);
  lodref_cpu_.reset(new int[XPU_MAX_LOD_SIZE_64]);
}

template <typename T, PrecisionType PType>
void SequenceExpandCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::SequenceExpandParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto* x = param.X;
  auto* y = param.Y;
  auto* out = param.Out;
  auto x_lod = x->lod();
  auto y_lod = y->lod();
  int ref_level = param.ref_level;
  if (ref_level == -1) {
    ref_level = y_lod.size() - 1;
  }

  auto* x_data = x->template data<T>();
  auto* out_data = out->template mutable_data<T>(TARGET(kXPU));

  if (y_lod[ref_level].size() <= 1 ||
      (y_lod[ref_level].size() == 2 && y_lod[ref_level][1] == 1)) {
    int r = xdnn::copy<int8_t>(ctx.GetRawContext(),
                               reinterpret_cast<const int8_t*>(x_data),
                               reinterpret_cast<int8_t*>(out_data),
                               x->numel() * sizeof(T));
    CHECK_EQ(r, 0) << "seqence_expand do copy failed.";
    return;
  }

  int dims = x->numel() / x->dims()[0];
  std::vector<uint64_t> ref_y_lod = y_lod[ref_level];
  // create ref_x_lod;
  std::vector<uint64_t> ref_x_lod;
  if (x->lod().size() == 1) {
    ref_x_lod = x->lod()[0];
  } else {
    ref_x_lod.resize(x->dims()[0] + 1);
    std::iota(ref_x_lod.begin(), ref_x_lod.end(), 0);
  }

  std::vector<int> ref_out_lod(ref_y_lod.size(), 0);
  std::vector<uint64_t> out_lod;
  out_lod.push_back(0);

  for (size_t i = 1; i < ref_y_lod.size(); ++i) {
    int repeat_num = ref_y_lod[i] - ref_y_lod[i - 1];
    int seq_len = ref_x_lod[i] - ref_x_lod[i - 1];
    for (int j = 0; j < repeat_num; ++j) {
      out_lod.push_back(out_lod.back() + seq_len);
    }
    ref_out_lod[i] = ref_out_lod[i - 1] + seq_len * repeat_num;
  }
  // write lod to out if x has lod
  if (x->lod().size()) {
    auto& ref_lod = *out->mutable_lod();
    ref_lod[0] = out_lod;
  }

  int lod_len = ref_y_lod.size();
  for (int i = 0; i < lod_len; ++i) {
    lodx_cpu_[i] = ref_x_lod[i];
    lodref_cpu_[i] = ref_y_lod[i];
    lody_cpu_[i] = ref_out_lod[i];
  }

  int r =
      xdnn::sequence_expand<float, int>(ctx.GetRawContext(),
                                        reinterpret_cast<const float*>(x_data),
                                        reinterpret_cast<float*>(out_data),
                                        {lodx_cpu_.get(), lod_len, nullptr},
                                        {lody_cpu_.get(), lod_len, nullptr},
                                        {lodref_cpu_.get(), lod_len, nullptr},
                                        dims);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using sequence_expand_float32 =
    paddle::lite::kernels::xpu::SequenceExpandCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    sequence_expand, kXPU, kFloat, kNCHW, sequence_expand_float32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
