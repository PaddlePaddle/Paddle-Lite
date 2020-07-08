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

#pragma once

/*
 * This file implements a light-weight OpDesc using NaiveBuffer.
 */

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "lite/model_parser/base/apis.h"
#include "lite/model_parser/naive_buffer/proto/framework.nb.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

/*
 * The lite::naive_buffer::OpDesc, an light-weight implementation of wrapper of
 * lite::naive_buffer::proto::OpDesc.
 */
class OpDesc : public OpDescAPI {
 public:
  using var_list_t = ListBuilder<proto::OpDesc::Var>;
  using str_list_t = ListBuilder<StringBuilder>;
  using attr_list_t = ListBuilder<proto::OpDesc::Attr>;

  OpDesc() = delete;

  explicit OpDesc(proto::OpDesc *desc) : desc_(desc) { CHECK(desc_); }

  void CopyFrom(OpDesc &op_desc) {
    CHECK(op_desc.Proto()) << "Source proto::OpDesc pointer can't be null";
    desc_ = op_desc.Proto();
  }

  proto::OpDesc *Proto() { return desc_; }

  const proto::OpDesc &ReadonlyProto() const { return *desc_; }

  std::string Type() const override {
    auto &builder = desc_->GetField<StringBuilder>("type");
    return builder.data();
  }

  void SetType(const std::string &type) override {
    auto *builder = desc_->GetMutableField<StringBuilder>("type");
    CHECK(builder);
    return builder->set(type);
  }

  // Get the arguments of parameter called `param`
  std::vector<std::string> Input(const std::string &param) const override {
    return GetArguments(desc_->GetField<var_list_t>("inputs"), param);
  }

  std::vector<std::string> InputArgumentNames() const override {
    return GetArgumentNames(desc_->GetField<var_list_t>("inputs"));
  }

  void SetInput(const std::string &param,
                const std::vector<std::string> &args) override {
    SetArgument(desc_->GetMutableField<var_list_t>("inputs"), param, args);
  }

  std::vector<std::string> Output(const std::string &param) const override {
    return GetArguments(desc_->GetField<var_list_t>("outputs"), param);
  }

  std::vector<std::string> OutputArgumentNames() const override {
    return GetArgumentNames(desc_->GetField<var_list_t>("outputs"));
  }

  void SetOutput(const std::string &param,
                 const std::vector<std::string> &args) override {
    SetArgument(desc_->GetMutableField<var_list_t>("outputs"), param, args);
  }

  bool HasAttr(const std::string &name) const override {
    const auto &xs = desc_->GetField<attr_list_t>("attrs");
    auto it =
        std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc::Attr &x) {
          auto &builder = x.GetField<StringBuilder>("name");
          return builder.data() == name;
        });
    return it != xs.end();
  }

  AttrType GetAttrType(const std::string &name) const override {
    const auto &xs = desc_->GetField<attr_list_t>("attrs");
    auto it =
        std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc::Attr &x) {
          auto &builder = x.GetField<StringBuilder>("name");
          return builder.data() == name;
        });
    CHECK(it != xs.end());
#define DEF_ONE(type__)                 \
  case proto::OpDesc::AttrType::type__: \
    return AttrType::type__;

    auto &builder = it->GetField<EnumBuilder<proto::OpDesc::AttrType>>("type");
    switch (builder.data()) {
      DEF_ONE(INT);
      DEF_ONE(FLOAT);
      DEF_ONE(STRING);
      DEF_ONE(INTS);
      DEF_ONE(FLOATS);
      DEF_ONE(STRINGS);
      DEF_ONE(BOOLEAN);
      DEF_ONE(BOOLEANS);
      DEF_ONE(BLOCK);
      DEF_ONE(LONG);
      DEF_ONE(BLOCKS);
      DEF_ONE(LONGS);
      default:
        LOG(FATAL) << "Unknown attribute type";
        return static_cast<AttrType>(-1);
    }
#undef DEF_ONE
  }

  std::vector<std::string> AttrNames() const override {
    std::vector<std::string> res;
    const auto &xs = desc_->GetField<attr_list_t>("attrs");
    std::transform(xs.begin(),
                   xs.end(),
                   std::back_inserter(res),
                   [](const proto::OpDesc::Attr &x) {
                     auto &builder = x.GetField<StringBuilder>("name");
                     return builder.data();
                   });
    return res;
  }

  template <typename T>
  void SetAttr(const std::string &name, const T &v);

  template <typename T>
  T GetAttr(const std::string &name) const;

  std::string DebugString() const { return "Not Implemented"; }

 private:
  std::vector<std::string> GetArguments(const var_list_t &xs,
                                        const std::string &param) const {
    std::vector<std::string> res;
    auto it =
        std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc::Var &it) {
          auto &builder = it.GetField<StringBuilder>("parameter");
          return builder.data() == param;
        });
    CHECK(it != xs.end());

    auto &list_builder = it->GetField<str_list_t>("arguments");
    std::transform(list_builder.begin(),
                   list_builder.end(),
                   std::back_inserter(res),
                   [](const StringBuilder &x) { return x.data(); });
    return res;
  }

  void SetArgument(var_list_t *xs,
                   const std::string &param,
                   const std::vector<std::string> &args) {
    auto it =
        std::find_if(xs->begin(), xs->end(), [&](const proto::OpDesc::Var &it) {
          auto &builder = it.GetField<StringBuilder>("parameter");
          return builder.data() == param;
        });
    if (it == xs->end()) {
      auto *new_arg = xs->New();
      auto *param_builder =
          new_arg->GetMutableField<StringBuilder>("parameter");
      CHECK(param_builder);
      param_builder->set(param);

      auto *arg_builder = new_arg->GetMutableField<str_list_t>("arguments");
      CHECK(arg_builder);
      for (const auto &arg : args) {
        arg_builder->New()->set(arg);
      }
    } else {
      auto *arg_builder = it->GetMutableField<str_list_t>("arguments");
      CHECK(arg_builder);
      arg_builder->Clear();
      for (const auto &arg : args) {
        arg_builder->New()->set(arg);
      }
    }
  }

  std::vector<std::string> GetArgumentNames(const var_list_t &xs) const {
    std::vector<std::string> res;
    std::transform(xs.begin(),
                   xs.end(),
                   std::back_inserter(res),
                   [](const proto::OpDesc::Var &x) {
                     auto &builder = x.GetField<StringBuilder>("parameter");
                     return builder.data();
                   });
    return res;
  }

 private:
  // Don't owned by naive_buffer::OpDesc
  proto::OpDesc *desc_;
};

template <>
void OpDesc::SetAttr<std::string>(const std::string &name,
                                  const std::string &v);

template <>
void OpDesc::SetAttr<std::vector<int>>(const std::string &name,
                                       const std::vector<int> &v);

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
