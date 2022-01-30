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

#include "lite/kernels/host/merge_lod_tensor_compute.h"
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

struct CopyRange {
  size_t begin;
  size_t end;
};

using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;
LoDAndOffset GetSubLoDAndAbsoluteOffset(const LoD &lod,
                                        size_t start_idx,
                                        size_t end_idx,
                                        size_t start_level) {
  LoD sub_lod;
  for (size_t level_idx = start_level; level_idx < lod.size(); ++level_idx) {
    CHECK(start_idx <= end_idx);
    CHECK(end_idx < lod[level_idx].size());
    std::vector<uint64_t> level_lens;
    for (size_t i = start_idx; i < end_idx; ++i) {
      level_lens.push_back(lod[level_idx][i + 1] - lod[level_idx][i]);
    }
    sub_lod.emplace_back(level_lens);
    start_idx = lod[level_idx][start_idx];
    end_idx = lod[level_idx][end_idx];
  }
  return LoDAndOffset{sub_lod, {start_idx, end_idx}};
}

void AppendLoD(LoD *lod, const LoD &lod_length) {
  CHECK(lod->empty() || lod->size() == lod_length.size());
  if (lod->empty()) {
    for (size_t i = 0; i < lod_length.size(); ++i) {
      lod->emplace_back(std::vector<uint64_t>({0}));
    }
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto &level = (*lod)[i];
    for (auto len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

void MergeLodTensorCompute::Run() {
  auto &param = Param<operators::MergeLodTensorParam>();
  const lite::Tensor *x = param.x;
  const lite::Tensor *mask = param.mask;
  const lite::Tensor *in_true = param.in_true;
  const lite::Tensor *in_false = param.in_false;
  lite::Tensor *out = param.out;
  int level = param.level;
  CHECK(in_true->IsInitialized() || in_false->IsInitialized());
  auto &in_true_dim = in_true->dims();
  auto &in_false_dim = in_false->dims();

  // only merge the first dim
  int64_t batch_size = 0;
  std::vector<int64_t> out_shape;
  if (in_true->IsInitialized()) {
    batch_size += in_true->dims()[0];
  }
  if (in_false->IsInitialized()) {
    batch_size += in_false->dims()[0];
  }
  out_shape.push_back(batch_size);
  if (in_true->IsInitialized()) {
    for (int i = 1; i < in_true_dim.size(); i++) {
      out_shape.push_back(in_true_dim[i]);
    }
  } else {
    for (int i = 1; i < in_false_dim.size(); i++) {
      out_shape.push_back(in_false_dim[i]);
    }
  }
  out->Resize(out_shape);

  size_t base_num = static_cast<size_t>(out->numel() / batch_size);
  auto *out_data = out->mutable_data<float>();
  auto *out_lod = out->mutable_lod();
  out_lod->clear();
  auto &mask_dim = mask->dims();
  auto *mask_data = mask->data<bool>();
  memset(out_data, 0, out->numel() * sizeof(float));

  size_t out_offset = 0;
  size_t in_true_idx = 0;
  size_t in_false_idx = 0;
  for (size_t i = 0; i < static_cast<size_t>(mask_dim[0]); i++) {
    const Tensor *input = nullptr;
    size_t *in_idx = nullptr;
    if (static_cast<int>(mask_data[i]) == 0) {
      input = in_false;
      in_idx = &in_false_idx;
    } else {
      input = in_true;
      in_idx = &in_true_idx;
    }
    auto lod_and_offset =
        GetSubLoDAndAbsoluteOffset(input->lod(), *in_idx, (*in_idx) + 1, 0);
    auto &lod_length = lod_and_offset.first;
    AppendLoD(out_lod, lod_length);
    size_t start_offset = lod_and_offset.second.first;
    size_t end_offset = lod_and_offset.second.second;
    CHECK(end_offset >= start_offset);
    size_t len = end_offset - start_offset;
    if (len == 0) {
      continue;
    }
    auto *in_src = input->data<float>() + base_num * start_offset;
    auto *out_dest = out_data + base_num * out_offset;
    size_t copy_num = base_num * len * sizeof(float);
    memcpy(out_dest, in_src, copy_num);
    out_offset += len;
    (*in_idx) += 1;
  }

  for (size_t i = 0; i < level; i++) {
    out_lod->insert(out_lod->begin(), x->lod()[i]);
  }

  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(merge_lod_tensor,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::MergeLodTensorCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindInput("InTrue", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InFalse", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
