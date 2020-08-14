// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/model_parser/flatbuffers/op_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

template <>
std::string OpDescView::GetAttr<std::string>(const char* name) const {
  const auto& it = desc_->attrs()->LookupByKey(name);
  if (!it->s()) {
    return std::string();
  }
  return it->s()->str();
}

template <>
std::string OpDescView::GetAttr<std::string>(const std::string& name) const {
  return GetAttr<std::string>(name.c_str());
}

template <>
lite::VectorView<std::string, Flatbuffers>
OpDescView::GetAttr<std::vector<std::string>>(const char* name) const {
  const auto& it = desc_->attrs()->LookupByKey(name);
  CHECK(it) << "Attr " << name << "does not exist.";
  return VectorView<std::string>(it->strings());
}

template <>
lite::VectorView<std::string, Flatbuffers>
OpDescView::GetAttr<std::vector<std::string>>(const std::string& name) const {
  return GetAttr<std::vector<std::string>>(name.c_str());
}

#define GET_ATTR_IMPL(T, fb_f__)                                             \
  template <>                                                                \
  typename lite::OpDataTypeTrait<T, Flatbuffers>::RT OpDescView::GetAttr<T>( \
      const char* name) const {                                              \
    const auto& it = desc_->attrs()->LookupByKey(name);                      \
    return it->fb_f__();                                                     \
  }                                                                          \
  template <>                                                                \
  typename lite::OpDataTypeTrait<T, Flatbuffers>::RT OpDescView::GetAttr<T>( \
      const std::string& name) const {                                       \
    return GetAttr<T>(name.c_str());                                         \
  }

#define GET_ATTRS_IMPL(T, fb_f__)                                            \
  template <>                                                                \
  typename lite::OpDataTypeTrait<T, Flatbuffers>::RT OpDescView::GetAttr<T>( \
      const char* name) const {                                              \
    const auto& it = desc_->attrs()->LookupByKey(name);                      \
    return typename lite::OpDataTypeTrait<T, Flatbuffers>::RT(it->fb_f__()); \
  }                                                                          \
  template <>                                                                \
  typename lite::OpDataTypeTrait<T, Flatbuffers>::RT OpDescView::GetAttr<T>( \
      const std::string& name) const {                                       \
    return GetAttr<T>(name.c_str());                                         \
  }

GET_ATTR_IMPL(int32_t, i);
GET_ATTR_IMPL(int16_t, block_idx);
GET_ATTR_IMPL(float, f);
GET_ATTR_IMPL(bool, b);
GET_ATTR_IMPL(int64_t, l);
GET_ATTRS_IMPL(std::vector<int>, ints);
GET_ATTRS_IMPL(std::vector<float>, floats);
GET_ATTRS_IMPL(std::vector<int64_t>, longs);
#undef GET_ATTR_IMPL
#undef GET_ATTRS_IMPL

#define ATTR_IMPL(T, fb_f__)                                                \
  template <>                                                               \
  T OpDesc::GetAttr<T>(const std::string& name) const {                     \
    return (*GetKeyIterator(name, desc_->attrs))->fb_f__;                   \
  }                                                                         \
  template <>                                                               \
  void OpDesc::SetAttr<T>(const std::string& name, const T& v) {            \
    auto& p = *InsertPair(name,                                             \
                          std::move(std::unique_ptr<proto::OpDesc_::AttrT>( \
                              new proto::OpDesc_::AttrT())),                \
                          &(desc_->attrs));                                 \
    p->fb_f__ = v;                                                          \
    p->type = ConvertAttrType(OpDataTypeTrait<T>::AT);                      \
    SetKey(name, &p);                                                       \
  }
ATTR_IMPL(int32_t, i);
ATTR_IMPL(int16_t, block_idx);
ATTR_IMPL(float, f);
ATTR_IMPL(bool, b);
ATTR_IMPL(int64_t, l);
ATTR_IMPL(std::string, s);
ATTR_IMPL(std::vector<int>, ints);
ATTR_IMPL(std::vector<float>, floats);
ATTR_IMPL(std::vector<int64_t>, longs);
ATTR_IMPL(std::vector<std::string>, strings);
#undef GET_ATTRS_IMPL

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
