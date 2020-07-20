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

#include "lite/kernels/xpu/sequence_concat_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SequenceConcatCompute::PrepareForRun() {
  lod0_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  lod1_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);

  lod0_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
  lod1_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

template <typename T>
inline LoD ConcatLoD(const std::vector<lite::Tensor*>& xs,
                     std::vector<lite::Tensor>* xs_in_order) {
  std::vector<uint64_t> result;
  result.resize(xs[0]->lod()[0].size());

  for (size_t i = 1; i < result.size(); ++i) {
    size_t sum = 0;
    for (size_t j = 0; j < xs.size(); ++j) {
      auto& x_lod = xs[j]->lod()[0];
      if (x_lod[i - 1] < x_lod[i]) {
        xs_in_order->emplace_back(xs[j]->Slice<T>(x_lod[i - 1], x_lod[i]));
      }
      sum += x_lod[i];
    }
    result[i] = sum;
  }
  LoD lod;
  lod.emplace_back(result);
  return lod;
}

void SequenceConcatCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto xs = param.X;
  auto out = param.Out;

  size_t lod_size = 0;
  for (auto& x : xs) {
    if (lod_size == 0) {
      lod_size = x->lod()[0].size();
    } else {
      CHECK_EQ(lod_size, x->lod()[0].size())
          << "The number of sequence must be same between each input";
    }
  }
  CHECK_NE(lod_size, 0) << "Each input must have sequence information";

  // TODO(miaotianxiang):
  int64_t dim0 = 0;
  int64_t feature_size = 0;
  std::vector<int64_t> out_dims;
  for (const auto& tensor : param.X) {
    const auto x_dims = tensor->dims();
    if (out_dims.empty()) {
      out_dims = x_dims.data();
    }
    dim0 += x_dims[0];
    if (feature_size == 0) {
      feature_size = x_dims.production() / x_dims[0];
    } else {
      CHECK_EQ(feature_size, x_dims.production() / x_dims[0])
          << "Inputs of sequence concat must have same feature size";
    }
  }
  out_dims[0] = dim0;
  out->Resize(out_dims);
  std::vector<lite::Tensor> x_in_order;
  out->set_lod(ConcatLoD<float>(xs, &x_in_order));

  CHECK(xs.size() == 2) << "XPU only support sequence_pool for 2 tensors";

  auto lod0 = xs[0]->lod()[0];
  auto lod1 = xs[1]->lod()[0];
  int batch_size = lod0.size() - 1;

  int* lod0_xpu = reinterpret_cast<int*>(lod0_xpu_guard_->addr_);
  int* lod1_xpu = reinterpret_cast<int*>(lod1_xpu_guard_->addr_);
  for (int i = 0; i < lod0.size(); ++i) {
    lod0_cpu[i] = lod0[i];
  }
  for (int i = 0; i < lod1.size(); ++i) {
    lod1_cpu[i] = lod1[i];
  }
  XPU_CALL(xpu_memcpy(lod0_xpu,
                      lod0_cpu.get(),
                      lod0.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(lod1_xpu,
                      lod1_cpu.get(),
                      lod1.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  int r = xdnn::sequence_concat(ctx.GetRawContext(),
                                xs[0]->data<float>(),
                                lod0_xpu,
                                xs[1]->data<float>(),
                                lod1_xpu,
                                out->mutable_data<float>(TARGET(kXPU)),
                                batch_size);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_concat,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SequenceConcatCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
