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

class XPUMMDNNBidEmbGrnnAttOp : public OpLite {
 public:
  XPUMMDNNBidEmbGrnnAttOp() {}

  explicit XPUMMDNNBidEmbGrnnAttOp(const std::string &op_type)
      : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMMDNNBidEmbGrnnAttOp"; }

 private:
  mutable XPUMMDNNBidEmbGrnnAttParam param_;
};

class XPUMMDNNBidEmbAttOp : public OpLite {
 public:
  XPUMMDNNBidEmbAttOp() {}

  explicit XPUMMDNNBidEmbAttOp(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMMDNNBidEmbAttOp"; }

 private:
  mutable XPUMMDNNBidEmbAttParam param_;
};

class XPUMMDNNMatchConvTopkOp : public OpLite {
 public:
  XPUMMDNNMatchConvTopkOp() {}

  explicit XPUMMDNNMatchConvTopkOp(const std::string &op_type)
      : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMMDNNMatchConvTopkOp"; }

 private:
  mutable XPUMMDNNMatchConvTopkParam param_;
};

class XPUMMDNNMergeAllOp : public OpLite {
 public:
  XPUMMDNNMergeAllOp() {}

  explicit XPUMMDNNMergeAllOp(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "XPUMMDNNMergeAllOp"; }

 private:
  mutable XPUMMDNNMergeAllParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
