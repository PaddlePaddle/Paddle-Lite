//
// Created by chenyaohuang on 2021/12/17.
//

#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class Unsqueeze2Pad3dSqueeze2Fuser : public FuseBase {
 public:
  explicit Unsqueeze2Pad3dSqueeze2Fuser(const std::string& unsqueeze2_type,
                                        const std::string& pad3d_type,
                                        const std::string& squeeze2_type) {
    pad3d_type_ = pad3d_type;
    squeeze2_type_ = squeeze2_type;
    unsqueeze2_type_ = unsqueeze2_type;
  }

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string pad3d_type_{"pad3d"};
  std::string squeeze2_type_{"squeeze2"};
  std::string unsqueeze2_type_{"unsqueeze2"};
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
