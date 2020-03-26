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

#include "lite/kernels/arm/distribute_fpn_proposals_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

const int kBoxDim = 4;

template <typename T>
static inline T BBoxArea(const T* box, bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

void DistributeFpnProposalsCompute::Run() {
  auto& param = Param<operators::DistributeFpnProposalsParam>();
  const lite::Tensor* fpn_rois = param.fpn_rois;
  std::vector<lite::Tensor*> multi_fpn_rois = param.multi_fpn_rois;
  lite::Tensor* restore_index = param.restore_index;
  int min_level = param.min_level;
  int max_level = param.max_level;
  int refer_level = param.refer_level;
  int refer_scale = param.refer_scale;
  int num_level = max_level - min_level + 1;

  CHECK_EQ(fpn_rois->lod().size(), 1);
  auto fpn_rois_lod = fpn_rois->lod().back();
  int fpn_rois_num = fpn_rois_lod[fpn_rois_lod.size() - 1];
  std::vector<int> target_level;
  // record the number of rois in each level
  std::vector<int> num_rois_level(num_level, 0);
  std::vector<int> num_rois_level_integral(num_level + 1, 0);
  for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
    auto fpn_rois_slice =
        fpn_rois->Slice<float>(static_cast<int64_t>(fpn_rois_lod[i]),
                               static_cast<int64_t>(fpn_rois_lod[i + 1]));
    const float* rois_data = fpn_rois_slice.data<float>();
    for (int j = 0; j < fpn_rois_slice.dims()[0]; ++j) {
      // get the target level of current rois
      float roi_scale = std::sqrt(BBoxArea(rois_data, false));
      int tgt_lvl =
          std::floor(log2(roi_scale / refer_scale + static_cast<float>(1e-6)) +
                     refer_level);
      tgt_lvl = std::min(max_level, std::max(tgt_lvl, min_level));
      target_level.push_back(tgt_lvl);
      num_rois_level[tgt_lvl - min_level]++;
      rois_data += kBoxDim;
    }
  }
  // define the output rois
  // pointer which point to each level fpn rois
  std::vector<float*> multi_fpn_rois_data(num_level);
  // lod0 which will record the offset information of each level rois
  std::vector<std::vector<uint64_t>> multi_fpn_rois_lod0;
  for (int i = 0; i < num_level; ++i) {
    // allocate memory for each level rois
    multi_fpn_rois[i]->Resize({num_rois_level[i], kBoxDim});
    multi_fpn_rois_data[i] = multi_fpn_rois[i]->mutable_data<float>();
    std::vector<uint64_t> lod0(1, 0);
    multi_fpn_rois_lod0.push_back(lod0);
    // statistic start point for each level rois
    num_rois_level_integral[i + 1] =
        num_rois_level_integral[i] + num_rois_level[i];
  }
  restore_index->Resize({fpn_rois_num, 1});
  int* restore_index_data = restore_index->mutable_data<int>();
  std::vector<int> restore_index_inter(fpn_rois_num, -1);
  // distribute the rois into different fpn level by target level
  for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
    Tensor fpn_rois_slice =
        fpn_rois->Slice<float>(static_cast<int64_t>(fpn_rois_lod[i]),
                               static_cast<int64_t>(fpn_rois_lod[i + 1]));
    const float* rois_data = fpn_rois_slice.data<float>();
    size_t cur_offset = fpn_rois_lod[i];
    // std::vector<size_t > lod_offset[num_level];
    for (int j = 0; j < num_level; j++) {
      multi_fpn_rois_lod0[j].push_back(multi_fpn_rois_lod0[j][i]);
    }
    for (int j = 0; j < fpn_rois_slice.dims()[0]; ++j) {
      int lvl = target_level[cur_offset + j];
      memcpy(multi_fpn_rois_data[lvl - min_level],
             rois_data,
             kBoxDim * sizeof(float));
      multi_fpn_rois_data[lvl - min_level] += kBoxDim;
      int index_in_shuffle = num_rois_level_integral[lvl - min_level] +
                             multi_fpn_rois_lod0[lvl - min_level][i + 1];
      restore_index_inter[index_in_shuffle] = cur_offset + j;
      multi_fpn_rois_lod0[lvl - min_level][i + 1]++;
      rois_data += kBoxDim;
    }
  }
  for (int i = 0; i < fpn_rois_num; ++i) {
    restore_index_data[restore_index_inter[i]] = i;
  }
  // merge lod information into LoDTensor
  for (int i = 0; i < num_level; ++i) {
    lite::LoD lod;
    lod.emplace_back(multi_fpn_rois_lod0[i]);
    multi_fpn_rois[i]->set_lod(lod);
  }
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(distribute_fpn_proposals,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::DistributeFpnProposalsCompute,
                     def)
    .BindInput("FpnRois", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("MultiFpnRois", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("RestoreIndex", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
