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

#include "lite/model_parser/naive_buffer/op_desc.h"
#include <set>
#include <utility>

namespace paddle {
namespace lite {
namespace naive_buffer {

proto::OpDesc::Attr* FindAttr(proto::OpDesc* desc, const std::string& name) {
  CHECK(desc);
  auto& xs = *desc->GetMutableField<ListBuilder<proto::OpDesc::Attr>>("attrs");
  auto it =
      std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc::Attr& x) {
        auto& builder = x.GetField<StringBuilder>("name");
        return builder.data() == name;
      });
  if (it == xs.end()) {
    auto* attr_builder = xs.New();
    auto* name_builder = attr_builder->GetMutableField<StringBuilder>("name");
    CHECK(name_builder);
    name_builder->set(name);
    return attr_builder;
  }
  return &(*it);
}

#define SET_ATTR_IMPL(T, ty__, bd__, pb_f__)                               \
  template <>                                                              \
  void OpDesc::SetAttr<T>(const std::string& name, const T& v) {           \
    auto* it = FindAttr(desc_, name);                                      \
    auto* type_builder =                                                   \
        it->GetMutableField<EnumBuilder<proto::OpDesc::AttrType>>("type"); \
    CHECK(type_builder);                                                   \
    type_builder->set(proto::OpDesc::AttrType::ty__);                      \
    auto* f_builder = it->GetMutableField<bd__##Builder>(#pb_f__);         \
    CHECK(f_builder);                                                      \
    f_builder->set(v);                                                     \
  }
SET_ATTR_IMPL(int, INT, Int32, i);
SET_ATTR_IMPL(float, FLOAT, Float32, f);
SET_ATTR_IMPL(bool, BOOLEAN, Bool, b);
SET_ATTR_IMPL(std::string, STRING, String, s);
SET_ATTR_IMPL(int64_t, LONG, Int64, l);
#undef SET_ATTR_IMPL

#define SET_ATTRS_IMPL(T, ty__, bd__, pb_f__)                              \
  template <>                                                              \
  void OpDesc::SetAttr<std::vector<T>>(const std::string& name,            \
                                       const std::vector<T>& v) {          \
    auto* it = FindAttr(desc_, name);                                      \
    auto* type_builder =                                                   \
        it->GetMutableField<EnumBuilder<proto::OpDesc::AttrType>>("type"); \
    CHECK(type_builder);                                                   \
    type_builder->set(proto::OpDesc::AttrType::ty__);                      \
    auto* vec_builder =                                                    \
        it->GetMutableField<ListBuilder<bd__##Builder>>(#pb_f__);          \
    CHECK(vec_builder);                                                    \
    vec_builder->Clear();                                                  \
    for (auto& i : v) {                                                    \
      auto* builder = vec_builder->New();                                  \
      builder->set(i);                                                     \
    }                                                                      \
  }
SET_ATTRS_IMPL(int, INTS, Int32, ints);
SET_ATTRS_IMPL(float, FLOATS, Float32, floats);
SET_ATTRS_IMPL(std::string, STRINGS, String, strings);
SET_ATTRS_IMPL(int64_t, LONGS, Int64, longs);
#undef SET_ATTRS_IMPL

const proto::OpDesc::Attr& GetFindAttr(const proto::OpDesc& desc,
                                       const std::string& name) {
  auto& xs = desc.GetField<ListBuilder<proto::OpDesc::Attr>>("attrs");
  auto it =
      std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc::Attr& x) {
        auto& builder = x.GetField<StringBuilder>("name");
        return builder.data() == name;
      });
  CHECK(it != xs.end());
  return *it;
}

#define GET_ATTR_IMPL(T, bd__, pb_f__)                   \
  template <>                                            \
  T OpDesc::GetAttr<T>(const std::string& name) const {  \
    auto& it = GetFindAttr(*desc_, name);                \
    auto& builder = it.GetField<bd__##Builder>(#pb_f__); \
    return builder.data();                               \
  }
GET_ATTR_IMPL(int32_t, Int32, i);
GET_ATTR_IMPL(int16_t, Int32, block_idx);
GET_ATTR_IMPL(float, Float32, f);
GET_ATTR_IMPL(bool, Bool, b);
GET_ATTR_IMPL(int64_t, Int64, l);
GET_ATTR_IMPL(std::string, String, s);
#undef GET_ATTR_IMPL

#define GET_ATTRS_IMPL(T, bd__, pb_f__)                                    \
  template <>                                                              \
  std::vector<T> OpDesc::GetAttr<std::vector<T>>(const std::string& name)  \
      const {                                                              \
    auto& it = GetFindAttr(*desc_, name);                                  \
    std::vector<T> res;                                                    \
    auto& list_builder = it.GetField<ListBuilder<bd__##Builder>>(#pb_f__); \
    for (size_t i = 0; i < list_builder.size(); ++i) {                     \
      res.push_back(list_builder.Get(i).data());                           \
    }                                                                      \
    return res;                                                            \
  }
GET_ATTRS_IMPL(int, Int32, ints);
GET_ATTRS_IMPL(float, Float32, floats);
GET_ATTRS_IMPL(std::string, String, strings);
GET_ATTRS_IMPL(int64_t, Int64, longs);
#undef GET_ATTRS_IMPL

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
