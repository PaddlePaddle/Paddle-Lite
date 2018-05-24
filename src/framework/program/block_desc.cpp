#include "block_desc.h"

namespace paddle_mobile {
namespace framework {

std::vector<std::shared_ptr<VarDesc>> BlockDesc::Vars() const {
  std::vector<std::shared_ptr<VarDesc>> res;
  for (const auto &p : vars_) {
    res.push_back(p.second);
  }
  return res;
}

std::vector<std::shared_ptr<OpDesc>> BlockDesc::Ops() const {
  std::vector<std::shared_ptr<OpDesc>> res;
  for (const auto &op : ops_) {
    res.push_back(op);
  }
  return res;
}

BlockDesc::BlockDesc(const proto::BlockDesc &desc):
        index_(desc.idx()), parent_index_(desc.parent_idx()) {
  for (const proto::VarDesc &var_desc : desc.vars()) {
    vars_[var_desc.name()].reset(new VarDesc(var_desc));
  }
  for (const proto::OpDesc &op_desc : desc.ops()) {
    ops_.emplace_back(new framework::OpDesc(op_desc));
  }
}

}  // namespace framework
}  // namespace paddle_mobile
