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

#pragma once
#include <string>
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace operators {

class XPUMmdnnBidEmbGrnnAttOp : public OpLite {
 public:
  XPUMmdnnBidEmbGrnnAttOp() {}

  explicit XPUMmdnnBidEmbGrnnAttOp(const std::string &op_type)
      : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMmdnnBidEmbGrnnAttOp"; }

 private:
  mutable XPUMmdnnBidEmbGrnnAttParam param_;
};

class XPUMmdnnBidEmbGrnnAttOp2 : public OpLite {
 public:
  XPUMmdnnBidEmbGrnnAttOp2() {}

  explicit XPUMmdnnBidEmbGrnnAttOp2(const std::string &op_type)
      : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override {
    return "XPUMmdnnBidEmbGrnnAttOp2";
  }

 private:
  mutable XPUMmdnnBidEmbGrnnAttParam2 param_;
};

class XPUMmdnnBidEmbAttOp : public OpLite {
 public:
  XPUMmdnnBidEmbAttOp() {}

  explicit XPUMmdnnBidEmbAttOp(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMmdnnBidEmbAttOp"; }

 private:
  mutable XPUMmdnnBidEmbAttParam param_;
};

class XPUMmdnnMatchConvTopkOp : public OpLite {
 public:
  XPUMmdnnMatchConvTopkOp() {}

  explicit XPUMmdnnMatchConvTopkOp(const std::string &op_type)
      : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMmdnnMatchConvTopkOp"; }

 private:
  mutable XPUMmdnnMatchConvTopkParam param_;
};

class XPUMmdnnMergeAllOp : public OpLite {
 public:
  XPUMmdnnMergeAllOp() {}

  explicit XPUMmdnnMergeAllOp(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMmdnnMergeAllOp"; }

 private:
  mutable XPUMmdnnMergeAllParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
