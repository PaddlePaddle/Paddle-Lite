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
#include <map>
#include <string>
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace subgraph {

const int FAILED = 1;
const int SUCCESS = 0;
const int REBUILD_WHEN_SHAPE_CHANGED = 2;
inline bool CHECK_FAILED(int status) { return status & FAILED; }
inline bool CHECK_SUCCESS(int status) { return !CHECK_FAILED(status); }
inline bool CHECK_REBUILD_WHEN_SHAPE_CHANGED(int status) {
  return status & REBUILD_WHEN_SHAPE_CHANGED;
}

using cvt_func_type =
    std::function<int(void* ctx, OpLite* op, KernelBase* kernel)>;
using cvt_map_type = std::map<int, std::map<std::string, cvt_func_type>>;
class SubgraphBridgeRegistry {
 public:
  static SubgraphBridgeRegistry& Instance();

  void Insert(const std::string& op_type,
              const TargetType& target,
              const cvt_func_type& cvt_func_name);
  const cvt_func_type& Select(const std::string& op_type,
                              const TargetType& target) const;
  bool Exists(const std::string& op_type, const TargetType& target) const;
  SubgraphBridgeRegistry() = default;

 private:
  cvt_map_type map_;
  DISALLOW_COPY_AND_ASSIGN(SubgraphBridgeRegistry);
};

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

#define STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(uniq_name, msg)         \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_SUBGRAPH_BRIDGE(op_type__, target__, cvt_func_name)      \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                          \
      __reg_subgraph_bridge_##op_type__##_##target__##__,                 \
      "REGISTER_SUBGRAPH_BRIDGE must be called in global namespace only " \
      "once!");                                                           \
  int __reg_subgraph_bridge_##op_type__##_##target__##_Insert() {         \
    paddle::lite::subgraph::SubgraphBridgeRegistry::Instance().Insert(    \
        #op_type__, TARGET(target__), cvt_func_name);                     \
    return 0;                                                             \
  }

#define USE_SUBGRAPH_BRIDGE(op_type__, target__, ...)                       \
  extern int __reg_subgraph_bridge_##op_type__##_##target__##_Insert();     \
  static int __reg_subgraph_bridge_##op_type__##_##target__##_Insert_return \
      UNUSED = __reg_subgraph_bridge_##op_type__##_##target__##_Insert();
