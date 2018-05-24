#include <vector>
#include <string>

#include "program_desc.h"

namespace paddle_mobile {
namespace framework {

ProgramDesc::ProgramDesc(const proto::ProgramDesc &desc) {
  for (auto &block_desc : desc.blocks()) {
    // new framework::BlockDesc(block_desc)
    blocks_.emplace_back(std::make_shared<BlockDesc>(block_desc));
  }
}

void ProgramDesc::Description(std::string header) {
#ifdef PADDLE_MOBILE_DEBUG
  if (header.size()){
    LOG(kLOG_INFO) << header;
  }
  for (const auto &block : this->blocks_) {
    LOG(kLOG_DEBUG) << "block: " << block->ID();
    LOG(kLOG_INFO) << "block ops size: " << block->Ops().size();
    for (int j = 0; j < block->Ops().size(); ++j) {
      const auto &op = block->Ops()[j];
      LOG(kLOG_DEBUG1) << "op: " << op->Type();
      for (auto &input : op->GetInputs()) {
        LOG(kLOG_DEBUG2) << "input parameter: " << input.first;
        for (auto &n : input.second) {
          LOG(kLOG_DEBUG3) << "argument - " << n;
        }
      }
      for (auto &output : op->GetOutputs()) {
        LOG(kLOG_DEBUG2) << "output parameter: " << output.first;
        for (auto &n : output.second) {
          LOG(kLOG_DEBUG3) << "argument - " << n;
        }
      }
      for (auto &attr : op->GetAttrMap()) {
        LOG(kLOG_DEBUG2) << "attr name:: " << attr.first;
        LOG(kLOG_DEBUG3) << "argument - " << attr.second;
      }
    }
  }
#endif
}

std::shared_ptr<BlockDesc> ProgramDesc::Block(size_t idx) {
  return blocks_[idx];
}

}  // namespace framework
}  // namespace paddle_mobile
