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

#include "lite/kernels/host/box_clip_compute.h"
#include <cmath>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void ClipTiledBoxes(const Tensor& im_info,
                    const Tensor& input_boxes,
                    Tensor* out) {
  T* out_data = out->mutable_data<T>();
  const T* im_info_data = im_info.data<T>();
  const T* input_boxes_data = input_boxes.data<T>();
  T zero(0);
  T im_w = round(im_info_data[1] / im_info_data[2]);
  T im_h = round(im_info_data[0] / im_info_data[2]);
  for (int64_t i = 0; i < input_boxes.numel(); ++i) {
    if (i % 4 == 0) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_w - 1), zero);
    } else if (i % 4 == 1) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_h - 1), zero);
    } else if (i % 4 == 2) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_w - 1), zero);
    } else {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_h - 1), zero);
    }
  }
}

void BoxClipCompute::Run() {
  auto& param = Param<operators::BoxClipParam>();
  const auto* input = param.Input;
  const auto* im_info = param.ImInfo;
  auto* output = param.Output;
  if (input->lod().size() > 1) {
    LOG(FATAL) << "Only support 0 and 1 level of LoD.";
  }

  auto box_lod = input->lod().back();
  // init output data
  auto* out_data = output->mutable_data<float>();
  memset(out_data, 0, sizeof(float) * output->numel());
  int64_t n = static_cast<int64_t>(box_lod.size() - 1);
  for (int i = 0; i < n; ++i) {
    Tensor im_info_slice = im_info->Slice<float>(i, i + 1);
    Tensor box_slice = input->Slice<float>(box_lod[i], box_lod[i + 1]);
    Tensor output_slice = output->Slice<float>(box_lod[i], box_lod[i + 1]);
    ClipTiledBoxes<float>(im_info_slice, box_slice, &output_slice);
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_clip,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::BoxClipCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("ImInfo", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
