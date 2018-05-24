#pragma once

#include <vector>

#include "common/types.h"
#include "framework/program/block_desc.h"
#include "framework/paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class ProgramDesc : PaddleMobileObject {
 public:
  friend class Node;
  friend class ProgramOptimize;
  explicit ProgramDesc(const proto::ProgramDesc &desc);
  std::shared_ptr<BlockDesc> Block(size_t idx);
  const std::vector<std::shared_ptr<BlockDesc>> &Blocks() { return blocks_; }
  ProgramDesc(const ProgramDesc &program_desc) {
    for (auto &block : program_desc.blocks_) {
      std::shared_ptr<BlockDesc> copy_block =
              std::make_shared<BlockDesc>(*block);
      blocks_.push_back(copy_block);
    }
  }

  void Description(std::string header = "");
 private:
  std::vector<std::shared_ptr<BlockDesc>> blocks_;
};

}  // namespace framework
}  // namespace paddle_mobile
