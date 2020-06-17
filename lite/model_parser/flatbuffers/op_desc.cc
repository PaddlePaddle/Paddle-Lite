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
std::string OpDesc::GetAttr<std::string>(const std::string& name) const {
  const auto& it = desc_->attrs()->LookupByKey(name.c_str());
  if (!it->s()) {
    return std::string();
  }
  return it->s()->str();
}

template <>
std::string OpDesc::GetAttr<std::string>(size_t idx) const {
  const auto& it = desc_->attrs()->Get(idx);
  if (!it->s()) {
    return std::string();
  }
  return it->s()->str();
}

template <>
std::vector<std::string> OpDesc::GetAttr<std::vector<std::string>>(
    const std::string& name) const {
  const auto& it = desc_->attrs()->LookupByKey(name.c_str());
  CHECK(it) << "Attr " << name << "does not exist.";
  std::vector<std::string> res;
  if (it->strings()) {
    res.reserve(it->strings()->size());
    for (const auto& v : *it->strings()) {
      res.push_back(v->str());
    }
  }
  return res;
}

template <>
std::vector<std::string> OpDesc::GetAttr<std::vector<std::string>>(
    size_t idx) const {
  const auto& it = desc_->attrs()->Get(idx);
  CHECK(it) << "Attr " << idx << "does not exist.";
  std::vector<std::string> res;
  if (it->strings()) {
    res.reserve(it->strings()->size());
    for (const auto& v : *it->strings()) {
      res.push_back(v->str());
    }
  }
  return res;
}

#define GET_ATTR_IMPL(T, fb_f__)                                \
  template <>                                                   \
  T OpDesc::GetAttr<T>(const std::string& name) const {         \
    const auto& it = desc_->attrs()->LookupByKey(name.c_str()); \
    return it->fb_f__();                                        \
  }                                                             \
  template <>                                                   \
  T OpDesc::GetAttr<T>(size_t idx) const {                      \
    const auto& it = desc_->attrs()->Get(idx);                  \
    return it->fb_f__();                                        \
  }

#define GET_ATTRS_IMPL(T, fb_f__)                               \
  template <>                                                   \
  T OpDesc::GetAttr<T>(const std::string& name) const {         \
    const auto& it = desc_->attrs()->LookupByKey(name.c_str()); \
    T res;                                                      \
    res.reserve(it->fb_f__()->size());                          \
    for (const auto& v : *it->fb_f__()) {                       \
      res.push_back(v);                                         \
    }                                                           \
    return res;                                                 \
  }                                                             \
  template <>                                                   \
  T OpDesc::GetAttr<T>(size_t idx) const {                      \
    const auto& it = desc_->attrs()->Get(idx);                  \
    T res;                                                      \
    res.reserve(it->fb_f__()->size());                          \
    for (const auto& v : *it->fb_f__()) {                       \
      res.push_back(v);                                         \
    }                                                           \
    return res;                                                 \
  }

GET_ATTR_IMPL(int32_t, i);
GET_ATTR_IMPL(int16_t, block_idx);
GET_ATTR_IMPL(float, f);
GET_ATTR_IMPL(bool, b);
GET_ATTR_IMPL(int64_t, l);
GET_ATTRS_IMPL(std::vector<int>, ints);
GET_ATTRS_IMPL(std::vector<float>, floats);
GET_ATTRS_IMPL(std::vector<int64_t>, longs);

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
