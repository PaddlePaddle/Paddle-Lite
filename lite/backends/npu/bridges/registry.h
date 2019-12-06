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
#include <utility>
#include <vector>
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridges {

// Type and registers of converters for converting Paddle Ops to HiAI IR graph
class cvt_ctx_type {
 public:
  template <typename T>
  std::shared_ptr<T> AddNode(std::string name) {
    auto unique_name = [&](const std::string& name) {
      int idx = 1;
      auto it = counts_.find(name);
      if (it == counts_.end()) {
        counts_.insert(std::make_pair(name, idx));
      } else {
        idx = ++(it->second);
      }
      return name + "_" + std::to_string(idx);
    };
    auto it = nodes_.find(name);
    if (it != nodes_.end()) {
      // Generate a new unique name as the key to bind the origin node:
      // new_name->node
      nodes_.insert(std::make_pair(unique_name(name + "_var"), it->second));
      nodes_.erase(it);
    }
    // Create a new node and bind with the name: name->new_node
    auto node = std::make_shared<T>(unique_name(name + "_op"));
    nodes_.insert(std::make_pair(name, node));
    return node;
  }

  std::shared_ptr<ge::Operator> GetNode(std::string name) {
    return nodes_.at(name);
  }

  bool HasNode(std::string name) { return nodes_.find(name) != nodes_.end(); }

 private:
  std::unordered_map<std::string, std::shared_ptr<ge::Operator>> nodes_;
  std::unordered_map<std::string, int> counts_;
};

const int FAILED = 1;
const int SUCCESS = 0;
const int REBUILD_WHEN_SHAPE_CHANGED = 2;
inline bool CHECK_FAILED(int status) { return status & FAILED; }
inline bool CHECK_SUCCESS(int status) { return !CHECK_FAILED(status); }
inline bool CHECK_REBUILD_WHEN_SHAPE_CHANGED(int status) {
  return status & REBUILD_WHEN_SHAPE_CHANGED;
}

using cvt_func_type = std::function<int(cvt_ctx_type* ctx, lite::OpLite* op)>;
using cvt_map_type = std::unordered_map<std::string, cvt_func_type>;
class Factory {
 public:
  static Factory& Instance();

  const cvt_map_type& AllFunctions() const { return map_; }
  bool HasType(const std::string& op_type) const;
  void Insert(const std::string& op_type, const cvt_func_type& cvt_func);
  Factory() = default;

 private:
  cvt_map_type map_;
  DISALLOW_COPY_AND_ASSIGN(Factory);
};

}  // namespace bridges
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
    paddle::lite::npu::bridges::Factory::Instance().Insert(#op_type,        \
                                                           cvt_func_name);  \
    return 0;                                                               \
  }

#define USE_NPU_BRIDGE(op_type)                                  \
  extern int __reg_npu_bridge_##op_type##_Insert();              \
  static int __reg_npu_bridge_##op_type##_Insert_return UNUSED = \
      __reg_npu_bridge_##op_type##_Insert();
