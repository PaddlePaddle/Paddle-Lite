//
// Created by chenyaohuang on 2021/12/17.
//

// unsqueeze2->pad3d->squeeze2 =>change to pad2d;
#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

class Unsqueeze2Pad3dSqueeze2FusePass : public ProgramPass {
public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
