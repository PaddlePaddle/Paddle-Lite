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

#include <vector>

#include "src/framework.pb-c.h"

namespace paddle_mobile {
namespace framework {

enum VarType_Type {
  VARTYPE_TYPE_BOOL = 0,
  VARTYPE_TYPE_INT16 = 1,
  VARTYPE_TYPE_INT32 = 2,
  VARTYPE_TYPE_INT64 = 3,
  VARTYPE_TYPE_FP16 = 4,
  VARTYPE_TYPE_FP32 = 5,
  VARTYPE_TYPE_FP64 = 6,
  VARTYPE_TYPE_LOD_TENSOR = 7,
  VARTYPE_TYPE_SELECTED_ROWS = 8,
  VARTYPE_TYPE_FEED_MINIBATCH = 9,
  VARTYPE_TYPE_FETCH_LIST = 10,
  VARTYPE_TYPE_STEP_SCOPES = 11,
  VARTYPE_TYPE_STEP_LOD_RANK_TABLE = 12,
  VARTYPE_TYPE_STEP_LOD_TENSOR_ARRAY = 13,
  VARTYPE_TYPE_STEP_PLACE_LIST = 14,
  VARTYPE_TYPE_READER = 15,
  VARTYPE_TYPE_CHANNEL = 16,
  VARTYPE_TYPE_RAW = 17,
  VARTYPE_TYPE_TUPLE = 18
};

class TensorDesc {
 public:
  TensorDesc() = default;
  TensorDesc(const TensorDesc &desc) {
    this->dims_ = desc.dims_;
    this->data_type_ = desc.data_type_;
  }

  explicit TensorDesc(
      PaddleMobile__Framework__Proto__VarType__TensorDesc *desc) {
    for (int i = 0; i < desc->n_dims; ++i) {
      int64_t d = desc->dims[i];
      dims_.emplace_back(d);
    }
    data_type_ = (VarType_Type)desc->data_type;
  }

  std::vector<int64_t> Dims() const { return dims_; }
  VarType_Type DataType() const { return data_type_; }

 private:
  std::vector<int64_t> dims_;
  VarType_Type data_type_;
};

}  // namespace framework
}  // namespace paddle_mobile
