#pragma once

#include <string>

#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"

namespace paddle_mobile {
namespace operators {

class FusionFcMatcher : public framework::FusionOpMatcher {
 public:
  FusionFcMatcher() {
    node_ = framework::Node("mul");
    node_ > std::make_shared<framework::Node>("elementwise_add");
  }

  void FolderNodes(framework::Node &node) {
    std::vector<std::shared_ptr<framework::OpDesc>> origin_descs = node.OpDescs(node_.Depth());
    node.Folder(node_.Depth(), Type(), {{"elementwise_add" , {"Y", "Z"}}});
  }

  std::string Type() {
    return "fc";
  }
};

class FusionFcOp {
 public:
 private:
};

static framework::FusionOpRegistrar fc_registrar(new FusionFcMatcher());

}  // namespace operators
}  // namespace paddle_mobile
