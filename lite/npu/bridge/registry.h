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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

// var_name, npu node point
using node_map_type =
    std::unordered_map<std::string, std::shared_ptr<ge::Operator>>;

using func_type = std::function<node_map_type(const std::shared_ptr<OpLite>,
                                              const node_map_type&)>;
using cvt_map_type = std::unordered_map<std::string, func_type>;
class Factory {
 public:
  static Factory& Instance();

  const cvt_map_type& AllFunctions() const { return map_; }
  bool HasType(const std::string& op_type) const;
  void Insert(const std::string& op_type, const func_type& func_name);
  Factory() = default;

 private:
  cvt_map_type map_;
  DISALLOW_COPY_AND_ASSIGN(Factory);
};

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

// some platform-independent defintion
#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

#define STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE(uniq_name, msg)              \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_NPU_BRIDGE(op_type, cvt_func_name)                         \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE(                                 \
      __reg_npu_bridge_##op_type##__,                                       \
      "REGISTER_NPU_BRIDGE must be called in global namespace only once!"); \
  int __reg_npu_bridge_##op_type##_Insert() {                               \
    paddle::lite::npu::bridge::Factory::Instance().Insert(#op_type,         \
                                                          cvt_func_name);   \
    return 0;                                                               \
  }

#define USE_NPU_BRIDGE(op_type)                                  \
  extern int __reg_npu_bridge_##op_type##_Insert();              \
  static int __reg_npu_bridge_##op_type##_Insert_return UNUSED = \
      __reg_npu_bridge_##op_type##_Insert();
