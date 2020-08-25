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

#include "lite/core/kernel.h"
#include <cstdlib>
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

std::string KernelBase::summary() const {
  STL::stringstream ss;
  ss << op_type() << ":" << TargetToStr(target()) << "/"
     << PrecisionToStr(precision()) << "/" << DataLayoutToStr(layout()) << "("
     << alias() << ")";
  return ss.str();
}

const Type *KernelBase::GetInputDeclType(const std::string &arg_name) const {
  CHECK(!op_type_.empty()) << "op_type should be set first";
  const auto *type = ParamTypeRegistry::Global().RetrieveInArgument(
      place(), GenParamTypeKey(), arg_name);
  CHECK(type) << "no type registered for kernel [" << op_type_
              << "] input argument [" << arg_name << "]"
              << " with key " << GenParamTypeKey();
  return type->type;
}

const Type *KernelBase::GetOutputDeclType(const std::string &arg_name) const {
  CHECK(!op_type_.empty()) << "op_type should be set first";
  const auto *type = ParamTypeRegistry::Global().RetrieveOutArgument(
      place(), GenParamTypeKey(), arg_name);
  CHECK(type) << "no type registered for kernel [" << GenParamTypeKey()
              << "] output argument [" << arg_name << "]";
  return type->type;
}

std::string KernelBase::GenParamTypeKey() const {
  STL::stringstream ss;
  ss << op_type() << "/" << alias_;
  return ss.str();
}

void KernelBase::ParseKernelType(const std::string &kernel_type,
                                 std::string *op_type,
                                 std::string *alias,
                                 Place *place) {
  auto parts = lite::SplitView(kernel_type, '/');
  CHECK_EQ(parts.size(), 5u);

  *op_type = parts[0];
  *alias = parts[1];

  const auto &target = parts[2];
  const auto &precision = parts[3];
  const auto &layout = parts[4];

  place->target = static_cast<TargetType>(target.to_digit<int>());
  place->precision = static_cast<PrecisionType>(precision.to_digit<int>());
  place->layout = static_cast<DataLayoutType>(layout.to_digit<int>());
}

std::string KernelBase::SerializeKernelType(const std::string &op_type,
                                            const std::string &alias,
                                            const Place &place) {
  STL::stringstream ss;
  ss << op_type << "/";
  ss << alias << "/";
  // We serialize the place value not the string representation here for
  // easier deserialization.
  ss << static_cast<int>(place.target) << "/";
  ss << static_cast<int>(place.precision) << "/";
  ss << static_cast<int>(place.layout);
  return ss.str();
}

bool ParamTypeRegistry::KeyCmp::operator()(
    const ParamTypeRegistry::key_t &a,
    const ParamTypeRegistry::key_t &b) const {
  return a.hash() < b.hash();
}

STL::ostream &operator<<(STL::ostream &os,
                         const ParamTypeRegistry::KernelIdTy &other) {
  std::string io_s = other.io == ParamTypeRegistry::IO::kInput ? "in" : "out";
  os << other.kernel_type << ":" << other.arg_name << ":" << io_s << ":"
     << other.place.DebugString();
  return os;
}

}  // namespace lite
}  // namespace paddle
