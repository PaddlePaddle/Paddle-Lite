//
// Created by liuRuiLong on 2018/5/4.
//

#include "op_desc.h"

namespace paddle_mobile{
namespace framework{

OpDesc::OpDesc(const proto::OpDesc &desc): desc_(desc) {
  for (int i = 0; i < desc_.inputs_size(); ++i) {
    const proto::OpDesc::Var &var = desc_.inputs(i);
    std::vector<std::string> &args = inputs_[var.parameter()];
    int arg_size = var.arguments_size();
    for (int j = 0; j < arg_size; ++j) {
        args.push_back(var.arguments(j));
    }
  }

  for (int i = 0; i < desc_.outputs_size(); ++i) {
    const proto::OpDesc::Var &var = desc_.outputs(i);
    std::vector<std::string> &args = outputs_[var.parameter()];
    int arg_size = var.arguments_size();
    for (int j = 0; j < arg_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }

  for (const proto::OpDesc::Attr &attr : desc_.attrs()) {
    std::string attr_name = attr.name();
    if (attr.type() != proto::AttrType::BLOCK) {
      attrs_[attr_name] = Attribute::GetAttrValue(attr);
//      if (attr.type() == proto::AttrType::INT){
//        std::cout << " attrName " << attr_name << " " << attrs_[attr_name].Get<int>() << std::endl;
//      }
    }
  }
}

const std::vector<std::string> &OpDesc::Input(const std::string &name) const{
  return inputs_.find(name)->second;
}

const std::vector<std::string> &OpDesc::Output(const std::string &name) const{
  return outputs_.find(name)->second;
}

Attribute OpDesc::GetAttr(const std::string &name) const{
  auto it = attrs_.find(name);
  return it->second;
}

const std::unordered_map<std::string, Attribute> &OpDesc::GetAttrMap() const{
  return attrs_;
}

}
}