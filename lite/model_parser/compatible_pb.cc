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

#include "lite/model_parser/compatible_pb.h"
#include <string>
#include <vector>

namespace paddle {
namespace lite {

template <typename OpDescType>
void InputsAnyToCpp(const OpDescType &any_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : any_desc.InputArgumentNames()) {
    cpp_desc->SetInput(param, any_desc.Input(param));
  }
}

template <typename OpDescType>
void InputsCppToAny(const cpp::OpDesc &cpp_desc, OpDescType *any_desc) {
  for (const std::string &param : cpp_desc.InputArgumentNames()) {
    any_desc->SetInput(param, cpp_desc.Input(param));
  }
}

template <typename OpDescType>
void OutputsAnyToCpp(const OpDescType &any_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : any_desc.OutputArgumentNames()) {
    cpp_desc->SetOutput(param, any_desc.Output(param));
  }
}

template <typename OpDescType>
void OutputsCppToAny(const cpp::OpDesc &cpp_desc, OpDescType *any_desc) {
  for (const std::string &param : cpp_desc.OutputArgumentNames()) {
    any_desc->SetOutput(param, cpp_desc.Output(param));
  }
}

template <typename OpDescType>
void AttrsAnyToCpp(const OpDescType &any_desc, cpp::OpDesc *cpp_desc) {
  using AttrType = OpDescAPI::AttrType;
  auto set_attr = [&](const std::string &name, AttrType type) {
    switch (type) {
      case AttrType::INT:
        cpp_desc->SetAttr<int32_t>(name,
                                   any_desc.template GetAttr<int32_t>(name));
        break;
      case AttrType::FLOAT:
        cpp_desc->SetAttr<float>(name, any_desc.template GetAttr<float>(name));
        break;
      case AttrType::STRING:
        cpp_desc->SetAttr<std::string>(
            name, any_desc.template GetAttr<std::string>(name));
        break;
      case AttrType::INTS:
        cpp_desc->SetAttr<std::vector<int>>(
            name, any_desc.template GetAttr<std::vector<int>>(name));
        break;
      case AttrType::FLOATS:
        cpp_desc->SetAttr<std::vector<float>>(
            name, any_desc.template GetAttr<std::vector<float>>(name));
        break;
      case AttrType::BOOLEAN:
        cpp_desc->SetAttr<bool>(name, any_desc.template GetAttr<bool>(name));
        break;
      case AttrType::STRINGS:
        cpp_desc->SetAttr<std::vector<std::string>>(
            name, any_desc.template GetAttr<std::vector<std::string>>(name));
        break;
      case AttrType::LONGS:
        cpp_desc->SetAttr<std::vector<int64_t>>(
            name, any_desc.template GetAttr<std::vector<int64_t>>(name));
        break;
      default:
        LOG(FATAL) << "Unsupported attr type found " << static_cast<int>(type);
    }
  };

  for (const auto &attr_name : any_desc.AttrNames()) {
    auto type = any_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

template <typename OpDescType>
void AttrsCppToAny(const cpp::OpDesc &cpp_desc, OpDescType *any_desc) {
  using AttrType = OpDescAPI::AttrType;
  auto set_attr = [&](const std::string &name, AttrType type) {
    switch (type) {
#define IMPL_ONE(type__, T)                                         \
  case AttrType::type__:                                            \
    any_desc->template SetAttr<T>(name, cpp_desc.GetAttr<T>(name)); \
    break;
      IMPL_ONE(INT, int32_t);
      IMPL_ONE(FLOAT, float);
      IMPL_ONE(STRING, std::string);
      IMPL_ONE(STRINGS, std::vector<std::string>);
      IMPL_ONE(FLOATS, std::vector<float>);
      IMPL_ONE(INTS, std::vector<int>);
      IMPL_ONE(BOOLEAN, bool);
      default:
        LOG(FATAL) << "Unsupported attr type found: " << static_cast<int>(type);
    }
  };
#undef IMPL_ONE
  for (const auto &attr_name : cpp_desc.AttrNames()) {
    auto type = cpp_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

#define TRANS_ANY_TO_CPP_IMPL(T)                                              \
  template <>                                                                 \
  void TransformOpDescAnyToCpp<T>(const T &any_desc, cpp::OpDesc *cpp_desc) { \
    cpp_desc->SetType(any_desc.Type());                                       \
    InputsAnyToCpp<T>(any_desc, cpp_desc);                                    \
    OutputsAnyToCpp<T>(any_desc, cpp_desc);                                   \
    AttrsAnyToCpp<T>(any_desc, cpp_desc);                                     \
  }
TRANS_ANY_TO_CPP_IMPL(pb::OpDesc);
TRANS_ANY_TO_CPP_IMPL(naive_buffer::OpDesc);
#undef TRANS_ANY_TO_CPP_IMPL

#define TRANS_CPP_TO_ANY_IMPL(T)                                              \
  template <>                                                                 \
  void TransformOpDescCppToAny<T>(const cpp::OpDesc &cpp_desc, T *any_desc) { \
    any_desc->SetType(cpp_desc.Type());                                       \
    InputsCppToAny<T>(cpp_desc, any_desc);                                    \
    OutputsCppToAny<T>(cpp_desc, any_desc);                                   \
    AttrsCppToAny<T>(cpp_desc, any_desc);                                     \
  }
TRANS_CPP_TO_ANY_IMPL(pb::OpDesc);
TRANS_CPP_TO_ANY_IMPL(naive_buffer::OpDesc);
#undef TRANS_CPP_TO_ANY_IMPL

}  // namespace lite
}  // namespace paddle
