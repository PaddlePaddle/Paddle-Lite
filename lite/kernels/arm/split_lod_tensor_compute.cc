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

#include "lite/kernels/arm/split_lod_tensor_compute.h"
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

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
    std::vector<size_t> level_lens;
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
      lod->emplace_back(1, 0);  // size = 1, value = 0;
    }
    *lod = LoD(lod_length.size(), std::vector<size_t>({0}));
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto &level = (*lod)[i];
    for (size_t len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

void log_lod(const LoD &lod, std::string info) {
  LOG(INFO) << info;
  for (auto l : lod) {
    LOG(INFO) << "---";
    for (auto i : l) {
      LOG(INFO) << i;
    }
  }
}

void log_tensor_data(const Tensor *t, std::string info) {
  LOG(INFO) << info;
  auto t_data = t->data<float>();
  for (int i = 0; i < t->numel(); i++) {
    LOG(INFO) << t_data[i];
  }
}

void SplitLodTensorCompute::Run() {
  auto &param = Param<operators::SplitLodTensorParam>();
  const lite::Tensor *x = param.x;
  const lite::Tensor *mask = param.mask;
  lite::Tensor *out_true = param.out_true;
  lite::Tensor *out_false = param.out_false;
  int level = param.level;

  auto &x_lod = x->lod();
  auto &mask_dim = mask->dims();
  auto *mask_data = mask->data<float>();

  std::vector<std::vector<CopyRange>> copy_ranges(2);
  // set out_true/out_false lod
  for (size_t t = 0; t < 2; t++) {
    LoD *lod = nullptr;
    if (t == 0) {
      lod = out_false->mutable_lod();
    } else {
      lod = out_true->mutable_lod();
    }
    lod->clear();
    for (size_t i = 0; i < static_cast<size_t>(mask_dim[0]); i++) {
      if (static_cast<size_t>(mask_data[i]) == t) {
        size_t start_idx = i;
        auto lod_and_offset =
            GetSubLoDAndAbsoluteOffset(x_lod, start_idx, start_idx + 1, level);

        auto &lod_length = lod_and_offset.first;
        AppendLoD(lod, lod_length);

        size_t start_offset = lod_and_offset.second.first;
        size_t end_offset = lod_and_offset.second.second;
        copy_ranges[t].emplace_back(CopyRange{start_offset, end_offset});
      }
    }
  }

  for (size_t t = 0; t < 2; ++t) {
    Tensor *out;
    if (t == 0) {
      out = out_false;
    } else {
      out = out_true;
    }
    auto &ranges = copy_ranges[t];
    size_t height = std::accumulate(
        ranges.begin(), ranges.end(), 0UL, [](size_t a, const CopyRange &b) {
          return a + b.end - b.begin;
        });
    auto x_dim = x->dims();
    x_dim[0] = static_cast<int64_t>(height);
    out->Resize(x_dim);
    auto *x_data = x->data<float>();
    auto *out_data = out->mutable_data<float>();
    auto out_dim = out->dims();
    size_t base_num = static_cast<size_t>(out->numel() / out_dim[0]);
    size_t offset = 0;
    for (auto &each_range : ranges) {
      size_t len = each_range.end - each_range.begin;
      if (len == 0) {
        continue;
      }

      auto *x_from = x_data + base_num * each_range.begin;
      auto *out_dest = out_data + base_num * offset;
      size_t copy_num = base_num * len * sizeof(float);
      memcpy(out_dest, x_from, copy_num);
      offset += len;
    }
  }

  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(split_lod_tensor,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SplitLodTensorCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("OutTrue", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("OutFalse", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
