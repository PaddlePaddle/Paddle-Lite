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

#include "framework/framework.pb-c.h"
#include "framework/paddle_mobile_object.h"
#include "framework/program/tensor_desc.h"

namespace paddle_mobile {
namespace framework {

/*

PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__BOOL = 0,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT16 = 1,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT32 = 2,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT64 = 3,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP16 = 4,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP32 = 5,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP64 = 6,

        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__LOD_TENSOR = 7,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__SELECTED_ROWS = 8,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FEED_MINIBATCH = 9,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FETCH_LIST = 10,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__STEP_SCOPES = 11,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__LOD_RANK_TABLE = 12,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__LOD_TENSOR_ARRAY = 13,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__PLACE_LIST = 14,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__READER = 15,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__CHANNEL = 16,

        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__RAW = 17,
        PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__TUPLE = 18


                                                                 */

class VarDesc {
 public:
  VarDesc(const VarDesc &var_desc) {
    this->data_type_ = var_desc.data_type_;
    this->name_ = var_desc.name_;
    this->persistable_ = var_desc.persistable_;
    this->tensor_desc_ = var_desc.tensor_desc_;
    this->type_ = var_desc.type_;
    /*
     *
     *  std::string name_;
  bool persistable_;
  TensorDesc tensor_desc_;
  VarType_Type type_;
  VarType_Type data_type_;
     * */
  }
  VarDesc(PaddleMobile__Framework__Proto__VarDesc *desc) {
    type_ = (VarType_Type)desc->type->type;
    name_ = std::string(desc->name);
    persistable_ = (bool)desc->persistable;

    switch (type_) {
      case VARTYPE_TYPE_SELECTED_ROWS:
        tensor_desc_ = TensorDesc(desc->type->selected_rows);
        break;
      case VARTYPE_TYPE_LOD_TENSOR:
        tensor_desc_ = TensorDesc(desc->type->lod_tensor->tensor);
        break;
      case VARTYPE_TYPE_STEP_LOD_TENSOR_ARRAY:
        desc->type->tensor_array->tensor->data_type;
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

  //  const proto::VarType::ChannelDesc &channel_desc() const {
  //    switch (desc_.type().type()) {
  //      case proto::VarType::CHANNEL:
  //        return desc_.type().channel();
  //      default:
  //        break;
  //    }
  //  }

  //  proto::VarType::Type GetDataType() const {
  //    switch (desc_.type().type()) {
  //      case proto::VarType::CHANNEL:
  //        return channel_desc().data_type();
  //        break;
  //      default:
  //        return tensor_desc().data_type();
  //    }
  //  }

  //  template <typename T>
  //  std::vector<T> RepeatedToVector(
  //      const google::protobuf::RepeatedField<T> &repeated_field) const {
  //    std::vector<T> ret;
  //    ret.reserve(repeated_field.size());
  //    std::copy(repeated_field.begin(), repeated_field.end(),
  //              std::back_inserter(ret));
  //    return ret;
  //  }

  //  std::vector<int64_t> GetShape() const {
  //    return this->RepeatedToVector(tensor_desc().dims());
  //  }

 private:
  std::string name_;
  bool persistable_;
  TensorDesc tensor_desc_;
  VarType_Type type_;
  VarType_Type data_type_;
};

}  // namespace framework
}  // namespace paddle_mobile
