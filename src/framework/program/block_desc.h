#pragma once

#include "framework/framework.pb.h"
#include "framework/program/op_desc.h"
#include "framework/program/var_desc.h"
#include "framework/paddle_mobile_object.h"


namespace paddle_mobile {
namespace framework {

class BlockDesc : PaddleMobileObject {
 public:
  friend class Node;
  friend class ProgramOptimize;
  BlockDesc(const proto::BlockDesc &desc);

  BlockDesc(const BlockDesc &block_desc):
          index_(block_desc.index_),
          parent_index_(block_desc.parent_index_) {
    for (auto &op_desc : block_desc.ops_) {
      std::shared_ptr<OpDesc> copy_op_desc = std::make_shared<OpDesc>(*op_desc);
      ops_.push_back(copy_op_desc);
    }

    for (auto &var_desc : block_desc.vars_) {
      std::shared_ptr<VarDesc> copy_var_desc =
              std::make_shared<VarDesc>(*var_desc.second);
      vars_[var_desc.first] = copy_var_desc;
    }
  }

  const int &ID() const { return index_; }

  const int &Parent() const { return parent_index_; }

  bool operator==(const paddle_mobile::framework::BlockDesc &in_block) const {
    return this->ID() == in_block.ID() && this->Parent() == in_block.Parent();
  }

  bool operator<(const paddle_mobile::framework::BlockDesc &in_block) const {
    return this->ID() < in_block.ID() && this->Parent() < in_block.Parent();
  }

  std::vector<std::shared_ptr<VarDesc>> Vars() const;
  std::vector<std::shared_ptr<OpDesc>> Ops() const;

 private:
  int index_;
  int parent_index_;
  std::vector<std::shared_ptr<OpDesc>> ops_;
  std::unordered_map<std::string, std::shared_ptr<VarDesc>> vars_;
};

}  // namespace framework
}  // namespace paddle_mobile

namespace std {

template <>
struct hash<paddle_mobile::framework::BlockDesc> {
  typedef paddle_mobile::framework::BlockDesc argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type const &s) const noexcept {
    result_type const h1(std::hash<int>{}(s.ID()));
    result_type const h2(std::hash<int>{}(s.ID()));
    return h1 ^ (h2 << 1);
  }
};

}  // namespace std
