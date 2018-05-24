#pragma once

#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"

namespace paddle_mobile {
namespace operators {

class FushionConvAddReluOpMatcher : public framework::FusionOpMatcher {
 public:
  FushionConvAddReluOpMatcher() {
    node_ = framework::Node("conv2d");
    node_ > std::make_shared<framework::Node>("elementwise_add") > std::make_shared<framework::Node>("relu");
  }

  void FolderNodes(framework::Node &node) {
    std::vector<std::shared_ptr<framework::OpDesc>> origin_descs = node.OpDescs(node_.Depth());
    node.Folder(node_.Depth(), Type(), {{"elementwise_add" , {"Y", "Z"}}});
  }

  std::string Type() {
    return "FusionConvAddRelu";
  }
};

class FusionFcOp {
 public:
 private:
};

static framework::FusionOpRegistrar fc_registrar(new FushionConvAddReluOpMatcher());

}
}
