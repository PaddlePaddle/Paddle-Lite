#pragma once

#include <string>
#include <vector>

#include "framework/operator.h"
#include "framework/program/program_desc.h"
#include "node.h"

namespace paddle_mobile {

namespace framework {
class ProgramOptimize {
 public:
  ProgramOptimize() {}
  std::shared_ptr<ProgramDesc> Optimize();
  std::shared_ptr<ProgramDesc> FushionOptimize(
      std::shared_ptr<ProgramDesc> ori_des);
 private:
  //                std::shared_ptr<ProgramDesc> ori_desc_;
  std::vector<std::unordered_map<std::string, std::shared_ptr<Node>>>
      outputs_nodes_;
};
}  // namespace framework
}  // namespace paddle_mobile
