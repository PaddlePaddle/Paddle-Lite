/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once;

#include "common/type_define.h"
#include "framework/lod_tensor.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/variable.h"

namespace paddle_mobile {
namespace operators {

using namespace framework;

class OpParam : PaddleMobileObject {
public:
protected:
  template <typename T>
  static T *InputFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Input", inputs, scope);
  }

  template <typename T>
  static T *InputXFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("X", inputs, scope);
  }

  template <typename T>
  static T *InputYFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Y", inputs, scope);
  }

  template <typename T>
  static std::vector<T *> InputMultiFrom(const VariableNameMap &inputs,
                                         const Scope &scope) {
    return GetMultiVarValue<T>("Input", inputs, scope);
  }

  template <typename T>
  static T *OutputFrom(const VariableNameMap &outputs, const Scope &scope) {
    return GetVarValue<T>("Output", outputs, scope);
  }

  template <typename T>
  static T *OutFrom(const VariableNameMap &outputs, const Scope &scope) {
    return GetVarValue<T>("Out", outputs, scope);
  }

  template <typename T>
  static T *FilterFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Filter", inputs, scope);
  }

  template <typename T>
  static const T GetAttr(std::string key, const AttributeMap &map) {
    return ((Attribute)map.at(key)).Get<T>();
  }

  template <typename T>
  static T *GetVarValue(std::string key, const VariableNameMap &var_map,
                        const Scope &scope) {
    auto var_vec = var_map.at(key);
    if (var_vec.size()) {
      //      std::cout << " get var value -- " << var_vec[0] << std::endl;
      auto var = scope.FindVar(var_vec[0]);
      return var->GetMutable<T>();
    } else {
      return nullptr;
    }
  }

  template <typename T>
  static std::vector<T *> GetMultiVarValue(std::string key,
                                           const VariableNameMap &var_map,
                                           const Scope &scope) {
    auto var_vecs = var_map.at(key);
    assert(var_vecs.size() > 1);
    std::vector<T *> var_res;
    for (auto &var_vec : var_vecs) {
      auto var = scope.FindVar(var_vec);
      var_res.push_back(var->GetMutable<T>());
    }
    return var_res;
  }
};

class ConvParam : OpParam {
public:
  ConvParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const framework::AttributeMap &attrs,
            const framework::Scope &scope) {
    filter_ = FilterFrom<framework::LoDTensor>(inputs, scope);
    input_ = InputFrom<framework::Tensor>(inputs, scope);
    output_ = OutputFrom<framework::Tensor>(outputs, scope);
    strides_ = GetAttr<std::vector<int>>("strides", attrs);
    paddings_ = GetAttr<std::vector<int>>("paddings", attrs);
    dilations_ = GetAttr<std::vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
  }

  const Tensor *Input() const { return input_; }

  const LoDTensor *Filter() const { return filter_; }

  Tensor *Output() const { return output_; }

  const std::vector<int> &Strides() const { return strides_; }

  const std::vector<int> &Paddings() const { return paddings_; }

  const std::vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

private:
  Tensor *input_;
  Tensor *output_;
  LoDTensor *filter_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int groups;
};

std::ostream &operator<<(std::ostream &os, const ConvParam &conv_param);

class ElementwiseAddParam : OpParam {
public:
  ElementwiseAddParam(const VariableNameMap &inputs,
                      const VariableNameMap &outputs,
                      const framework::AttributeMap &attrs,
                      const framework::Scope &scope) {
    input_x_ = InputXFrom<framework::Tensor>(inputs, scope);
    input_y_ = InputYFrom<framework::Tensor>(inputs, scope);
    out_ = OutFrom<framework::Tensor>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputY() const { return input_y_; }

  Tensor *Out() const { return out_; }

  const int &Axis() const { return axis_; }

private:
  Tensor *input_x_;
  Tensor *input_y_;
  Tensor *out_;
  int axis_;
};

class MulParam : OpParam {
public:
  MulParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
           const framework::AttributeMap &attrs,
           const framework::Scope &scope) {
    input_x_ = InputXFrom<framework::Tensor>(inputs, scope);
    input_y_ = InputYFrom<framework::Tensor>(inputs, scope);
    out_ = OutFrom<framework::Tensor>(outputs, scope);
    x_num_col_dims_ = GetAttr<int>("x_num_col_dims", attrs);
    y_num_col_dims_ = GetAttr<int>("y_num_col_dims", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputY() const { return input_y_; }

  Tensor *Out() const { return out_; }

  const int &XNumColDims() const { return x_num_col_dims_; }

  const int &YNumColDims() const { return y_num_col_dims_; }

private:
  Tensor *input_x_;
  Tensor *input_y_;
  Tensor *out_;
  int x_num_col_dims_;
  int y_num_col_dims_;
};

class ConcatParam : public OpParam {
public:
  ConcatParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
              const framework::AttributeMap &attrs,
              const framework::Scope &scope) {
    inputs_ = InputMultiFrom<framework::Tensor>(inputs, scope);
    out_ = OutFrom<framework::Tensor>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
  }

  std::vector<Tensor *> Inputs() const { return inputs_; }

  Tensor *Out() const { return out_; }

  const int &Axis() const { return axis_; }

private:
  std::vector<Tensor *> inputs_;
  Tensor *out_;
  int axis_;
};

} // namespace operators
} // namespace paddle_mobile
