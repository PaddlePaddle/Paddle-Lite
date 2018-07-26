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

#include "src/framework.pb-c.h"
#include "src/tensor_desc.h"

namespace paddle_mobile {
namespace framework {

class VarDesc {
 public:
  VarDesc(const VarDesc &var_desc) {
    this->data_type_ = var_desc.data_type_;
    this->name_ = var_desc.name_;
    this->persistable_ = var_desc.persistable_;
    this->tensor_desc_ = var_desc.tensor_desc_;
    this->type_ = var_desc.type_;
  }
  explicit VarDesc(PaddleMobile__Framework__Proto__VarDesc *desc) {
    type_ = (VarType_Type)desc->type->type;
    name_ = std::string(desc->name);
    persistable_ = static_cast<bool>(desc->persistable);

    switch (type_) {
      case VARTYPE_TYPE_SELECTED_ROWS:
        tensor_desc_ = TensorDesc(desc->type->selected_rows);
        break;
      case VARTYPE_TYPE_LOD_TENSOR:
        tensor_desc_ = TensorDesc(desc->type->lod_tensor->tensor);
        break;
      case VARTYPE_TYPE_STEP_LOD_TENSOR_ARRAY:
        // desc->type->tensor_array->tensor->data_type;
        tensor_desc_ = TensorDesc(desc->type->tensor_array->tensor);

        break;
      default:
        break;
    }
    switch (type_) {
      case VARTYPE_TYPE_CHANNEL:
        data_type_ = (VarType_Type)desc->type->channel->data_type;
        break;
      default:
        data_type_ = tensor_desc_.DataType();
        break;
    }
  }
  std::string Name() const { return name_; }

  VarType_Type Type() const { return type_; }

  bool Persistable() const { return persistable_; }

  const TensorDesc &Tensor_desc() const { return tensor_desc_; }

 private:
  std::string name_;
  bool persistable_;
  TensorDesc tensor_desc_;
  VarType_Type type_;
  VarType_Type data_type_;
};

}  // namespace framework
}  // namespace paddle_mobile
