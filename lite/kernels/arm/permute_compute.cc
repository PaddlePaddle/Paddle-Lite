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

#include "lite/kernels/arm/permute_compute.h"
#include <string>
#include <vector>
#include "lite/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void PermuteCompute::PrepareForRun() {
  // 1 、读入数据----我先需要inputs、outputs、param、
  auto& param = Param<operators::PermuteParam>();
  std::vector<lite::Tensor*> inputs = param.X;
  std::vector<lite::Tensor*> outputs = param.Out;
  // int *order_data= param.order;
  LOG(INFO) << "into prepare for run";
  _num_axes = inputs[0]->dims().size();
  _count = outputs[0]->dims().count(0, _num_axes);  // modified
  CHECK_EQ(inputs[0]->dims().size(), param.order.size())
      << "ERROR: permute order size is not match to input dims\n";
  // set _need_permute
  _need_permute = false;
  for (int i = 0; i < _num_axes; ++i) {
    if (param.order[i] != i) {
      _need_permute = true;
      break;
    }
  }
  if (!_need_permute) {
    LOG(INFO) << "need permute break";
    return;
  }
  //! for basic permute
  std::vector<int> axis_diff;
  int j = 0;
  for (int i = 0; i < _num_axes; ++i) {
    if (param.order[j] != i) {
      axis_diff.push_back(j);
      // LOG(INFO) << "diff axis: " << _order_dims[j];
    } else {
      j++;
    }
  }

  if (inputs[0]->dims().count(axis_diff[0], _num_axes) == 1) {
    _need_permute = false;
    LOG(INFO) << "don't need permute!";
    return;
  }

  if (axis_diff.size() == 1) {
    LOG(INFO) << "transpose true";
    _transpose = true;
    _trans_num = inputs[0]->dims().count(0, std::max(axis_diff[0], 0));
    _trans_w = inputs[0]->dims().count(axis_diff[0] + 1, _num_axes);
    _trans_h = inputs[0]->dims()[axis_diff[0]];
#ifdef ENABLE_DEBUG
    printf("permute: transpose=true, num= %d, h=%d, w=%d\n",
           _trans_num,
           _trans_h,
           _trans_w);
#endif
  } else {
    LOG(INFO) << "transpose false";
    _transpose = false;
    // added by zhiqiang,to release shape::get_stride函数
    DDimLite input_dims = inputs[0]->dims();
    _new_steps.resize(input_dims.size());
    for (int i = 0; i < input_dims.size(); ++i) {
      _new_steps[i] = input_dims.count(i + 1, input_dims.size());
    }
    _old_steps.resize(input_dims.size());
    for (int i = 0; i < input_dims.size(); ++i) {
      _old_steps[i] = input_dims.count(i + 1, input_dims.size());
    }

#ifdef ENABLE_DEBUG
    printf("permute: transpose=false\n");
#endif
  }
}

void PermuteCompute::Run() {
  // 1、读入数据
  auto& param = Param<operators::PermuteParam>();
  std::vector<lite::Tensor*> inputs = param.X;
  std::vector<lite::Tensor*> output = param.Out;
  // int *order_data= param.order;

  //! only copy the data
  // 这个不要了，一定要permute
  if (!_need_permute) {
    // output[0]=inputs[0];
    for (int i = 0; i < inputs[0]->dims().production(); i++) {
      output[0]->mutable_data<float>()[i] = inputs[0]->mutable_data<float>()[i];
    }
    return;
  }
  // 2、输入输出
  const float* din = inputs[0]->mutable_data<float>();
  float* dout = output[0]->mutable_data<float>();

  // 3、 transpose the data
  if (_transpose) {
    lite::arm::math::transpose_mat(din, dout, _trans_num, _trans_w, _trans_h);
  } else {
    lite::arm::math::permute_basic(_count,
                                   din,
                                   param.order.data(),
                                   _old_steps.data(),
                                   _new_steps.data(),
                                   _num_axes,
                                   dout);
  }

  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(permute,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::PermuteCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
