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

#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/* Detect Max Pooling which can pad zero instead of pad -inf    */
/* For example:                                                 */
/* graph[1]: sub block                                          */
/*                  relu/sigmoid/relu6...                       */
/*                       |                                      */
/*                       |                                      */
/*                   max_pooling                                */

class XPUPositiveActMaxPoolingFuser : public FuseBase {
 public:
  explicit XPUPositiveActMaxPoolingFuser(const std::string& act_type) {
    act_type_ = act_type;
  }
  void BuildPattern() override {
    auto* pre_op = OpNode("pre_op", act_type_);
    auto* pre_op_out = VarNode("pre_out")
                           ->assert_is_op_output(act_type_, "Out")
                           ->assert_is_op_input("pool2d", "X");
    auto* max_pool = OpNode("pool2d", "pool2d")
                         ->assert_op_attr<std::string>("pooling_type", "max");

    *pre_op >> *pre_op_out >> *max_pool;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* pool_instruct = matched.at("pool2d")->stmt();
    auto pool_op_desc = *pool_instruct->mutable_op_info();
    auto pool_op = pool_instruct->op();
    pool_op_desc.SetAttr<bool>("pad_zero", true);
    pool_instruct->ResetOp(pool_op_desc, pool_op->valid_places());
  }

 private:
  std::string act_type_;
};

}  // namespace fusion

class XPUMaxPoolingPadZeroDetectFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto act_type : {"relu", "sigmoid", "hard_sigmoid", "relu6"}) {
      fusion::XPUPositiveActMaxPoolingFuser fuser(act_type);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__max_pooling_pad_zero_detect_fuse_pass,
                  paddle::lite::mir::XPUMaxPoolingPadZeroDetectFusePass)
    .BindTargets({TARGET(kXPU)});
