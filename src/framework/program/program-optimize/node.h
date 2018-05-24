#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/log.h"
#include "framework/program/op_desc.h"
#include "framework/paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class Node : PaddleMobileObject {
 public:
  Node() {}
  explicit Node(const std::string &type) : type_(type) {}
  explicit Node(std::shared_ptr<OpDesc> op_desc)
      : op_desc_(op_desc), type_(op_desc->Type()) {}
  Node &operator>(std::shared_ptr<Node> node);
  bool operator==(const Node &in);
  std::string ToString() const;
  std::shared_ptr<Node> To(int size);
  uint Depth(uint begin = 0);
  Node &Folder(
      uint size, std::string type,
      std::map<std::string, std::pair<std::string, std::string>> change_map);
  std::vector<std::shared_ptr<framework::OpDesc>> OpDescs(uint size);
  std::vector<std::shared_ptr<framework::OpDesc>> OpDescs();
  void OpDescs(std::vector<std::shared_ptr<framework::OpDesc>> *op_desc,  Node *node);
  std::shared_ptr<framework::OpDesc> OpDesc() { return op_desc_; }
  std::string BeginType() { return type_; }
  void Description();

 private:
  void OpDescs(uint size,
               std::vector<std::shared_ptr<framework::OpDesc>> *op_desc);
  void To(int index, std::shared_ptr<Node>);
  void Folder(
      std::shared_ptr<framework::OpDesc> op_desc,
      std::vector<std::shared_ptr<Node>> *outputs, uint index,
      std::map<std::string, std::pair<std::string, std::string>> *change,
      Node *begin_node);
  std::shared_ptr<framework::OpDesc> op_desc_;
  std::string ToString(std::string blank, const Node *node) const;
  std::vector<std::shared_ptr<Node>> outputs_;
  std::vector<Node *> inputs_;
  std::string type_;
};

Print &operator<<(Print &printer, const Node &node);
}  // namespace framework
}  // namespace paddle_mobile
