#pragma once;

#include <map>
#include <string>
#include <vector>
#include "framework/attribute.h"
#include "framework/scope.h"

namespace paddle_mobile {

namespace framework {
template <typename Dtype>
class OperatorBase;
class OpDesc;
class BlockDesc;
class InferShapeContext;
}  // namespace framework

using VariableNameMap = std::map<std::string, std::vector<std::string>>;

template <typename Dtype>
using OpCreator = std::function<framework::OperatorBase<Dtype> *(
    const std::string & /*type*/, const VariableNameMap & /*inputs*/,
    const VariableNameMap & /*outputs*/,
    const framework::AttributeMap & /*attrs*/,
    std::shared_ptr<framework::Scope> /*scope*/)>;

using GradOpMakerFN =
    std::function<std::vector<std::unique_ptr<framework::OpDesc>>(
        const framework::OpDesc &,
        const std::unordered_set<std::string> & /*no_grad_set*/,
        std::unordered_map<std::string, std::string> * /*grad_to_var*/,
        const std::vector<framework::BlockDesc *> &grad_block)>;

using InferVarTypeFN = std::function<void(const framework::OpDesc & /*op_desc*/,
                                          framework::BlockDesc * /*block*/)>;

using InferShapeFN = std::function<void(framework::InferShapeContext *)>;
};  // namespace paddle_mobile
