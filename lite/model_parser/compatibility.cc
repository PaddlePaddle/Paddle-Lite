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
#include "lite/model_parser/compatibility.h"

#ifndef LITE_ON_TINY_PUBLISH
#include "lite/core/type_system.h"
#include "lite/model_parser/cpp_desc.h"
#include "lite/model_parser/naive_buffer/block_desc.h"
#include "lite/model_parser/naive_buffer/op_desc.h"
#include "lite/model_parser/naive_buffer/program_desc.h"
#include "lite/model_parser/naive_buffer/var_desc.h"
#endif

namespace paddle {
namespace lite {

template <typename T>
bool CompatibleChecker<T>::CheckKernelVersion(const std::string& type,
                                              const lite_api::Place& place) {
  int64_t impl_version = ParamTypeRegistry::Global().GetVersion(type, place);
  const int64_t prog_version = program_.Version();
  VLOG(3) << "Kernel implement version: " << type << ", " << impl_version;
  VLOG(3) << "Kernel program version: " << type << ", " << prog_version;
  if (impl_version == -1) {
    impl_version = mini_version_;
  }
  return prog_version <= impl_version;
}

template <typename T>
std::set<std::string> CompatibleChecker<T>::OpsType(T* program) {
  LOG(WARNING) << "OpsType() is not yet implemented.";
  return std::set<std::string>();
}

#ifndef LITE_ON_TINY_PUBLISH
template <>
std::set<std::string> CompatibleChecker<cpp::ProgramDesc>::OpsType(
    cpp::ProgramDesc* program) {
  std::set<std::string> ops_type;
  for (size_t i = 0; i < program->BlocksSize(); ++i) {
    auto* block = program->GetBlock<cpp::BlockDesc>(i);
    for (size_t j = 0; j < block->OpsSize(); ++j) {
      auto* op = block->GetOp<cpp::OpDesc>(j);
      ops_type.insert(op->Type());
    }
  }
  return ops_type;
}

template class CompatibleChecker<cpp::ProgramDesc>;
#endif  // LITE_ON_TINY_PUBLISH

}  // namespace lite
}  // namespace paddle
