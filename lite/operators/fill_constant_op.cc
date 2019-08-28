// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

class FillConstantOp : public OpLite {
 public:
  explicit FillConstantOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.Out);
    return true;
  }

  bool InferShape() const override {
    param_.Out->Resize(param_.shape);
    return true;
  }

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override {
    auto Out_name = opdesc.Output("Out").front();

    param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
    param_.dtype = opdesc.GetAttr<int>("dtype");
    param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");
    param_.value = opdesc.GetAttr<float>("value");
    param_.force_cpu = opdesc.GetAttr<bool>("force_cpu");
    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "fill_constant"; }

 private:
  mutable operators::FillConstantParam param_;
};

class FillConstantBatchLikeOp : public OpLite {
 public:
  explicit FillConstantBatchLikeOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.out);
    CHECK_OR_FALSE(param_.input);
    CHECK_GT_OR_FALSE(param_.shape.size(), 0);
    CHECK_GE_OR_FALSE(param_.input_dim_idx, 0);
    CHECK_GE_OR_FALSE(param_.output_dim_idx, 0);
    return true;
  }

  bool InferShape() const override {
    auto output_dim = param_.shape;
    output_dim[param_.output_dim_idx] =
        param_.input->dims()[param_.input_dim_idx];
    param_.out->Resize(output_dim);
    return true;
  }

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override {
    auto Out_name = opdesc.Output("Out").front();
    auto In_name = opdesc.Input("Input").front();

    param_.out = GetMutableVar<lite::Tensor>(scope, Out_name);
    param_.input = GetMutableVar<lite::Tensor>(scope, In_name);
    param_.dtype = opdesc.GetAttr<int>("dtype");
    param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");
    if (opdesc.HasAttr("value")) {
      param_.value = opdesc.GetAttr<float>("value");
    }
    if (opdesc.HasAttr("input_dim_idx")) {
      param_.input_dim_idx = opdesc.GetAttr<int>("input_dim_idx");
    }
    if (opdesc.HasAttr("output_dim_idx")) {
      param_.output_dim_idx = opdesc.GetAttr<int>("output_dim_idx");
    }

    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override {
    return "fill_constant_batch_size_like";
  }

 private:
  mutable operators::FillConstantBatchLikeParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fill_constant, paddle::lite::operators::FillConstantOp);
REGISTER_LITE_OP(fill_constant_batch_size_like,
                 paddle::lite::operators::FillConstantBatchLikeOp);
