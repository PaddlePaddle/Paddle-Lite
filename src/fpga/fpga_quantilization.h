/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <string>
#include "common/types.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/scope.h"
#include "framework/tensor.h"

namespace paddle_mobile {

bool is_conv(std::string type) {
  if (type.compare(G_OP_TYPE_CONV) == 0) {
    return true;
  }
  if (type.compare(G_OP_TYPE_FUSION_CONV_ADD) == 0) {
    return true;
  }
  if (type.compare(G_OP_TYPE_FUSION_CONV_ADD_RELU) == 0) {
    return true;
  }
  if (type.compare(G_OP_TYPE_FUSION_CONV_BN_RELU) == 0) {
    return true;
  }
  if (type.compare(G_OP_TYPE_FUSION_CONV_ADD_BN) == 0) {
    return true;
  }
  return false;
}

template <typename Dtype>
void quantilize_op(std::shared_ptr<framework::OperatorBase<Dtype>> op,
                   std::shared_ptr<framework::Scope> scope) {
  if (!is_conv(op.get()->Type())) {
    return;
  }
  framework::Tensor* filter = nullptr;
  auto var_vec = op.get()->Inputs().at("Filter");
  if (!var_vec.empty()) {
    auto var = scope.get()->FindVar(var_vec[0]);
    filter = var->template GetMutable<framework::LoDTensor>();
  }
  float scale = 0;

  // 32bit filter -> 8bit filter;
  if (filter->type() == typeid(float)) {
    framework::Tensor* originalFilter = filter;
    framework::Tensor* quantFilter = new framework::Tensor();
    float* floatData = originalFilter->data<float>();
    int8_t* intData = quantFilter->mutable_data<int8_t>();
  }
}

}  // namespace paddle_mobile
