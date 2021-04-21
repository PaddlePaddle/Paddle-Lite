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

#include "lite/core/mir/node.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

const OpInfo *mir::Node::Stmt::op_info() const {
  CHECK(op_);
  return op_->op_info();
}

Place mir::Node::Stmt::place() const {
  CHECK(!valid_kernels_.empty());
  return valid_kernels_.front()->place();
}

KernelBase &mir::Node::Stmt::picked_kernel() {
// Error message: if current kernel is not supported, WITH_EXTRA lib is
// suggested.
#ifndef LITE_BUILD_EXTRA
  std::string error_message =
      "Error: Please use Paddle-Lite lib with all ops, which is marked with "
      "`with_extra`. Current lib is of tiny_publish, in which only basic "
      "kernels are included and we can not create kernel for '"
      << op_type()
      << "'.\n Two ways are suggested to get Paddle-Lite lib with all ops:\n   "
         " 1. Download pre-commit lib which is marked with `with_extra`.\n    "
         "2. Compile Paddle-Lite with command `--with_extra=ON`.";
#else
  std::string error_message =
      "Error: This model is not supported, because kernel for '" + op_type() +
      "' is not supported by Paddle-Lite.";
#endif
  CHECK(!valid_kernels_.empty()) << error_message;
  return *valid_kernels_.front();
}

OpInfo *mir::Node::Stmt::mutable_op_info() {
  CHECK(op_);
  return op_->mutable_op_info();
}

void mir::Node::Stmt::ResetOp(const cpp::OpDesc &op_desc,
                              const std::vector<Place> &valid_places,
                              lite::Scope *scope) {
  CHECK((op_ && op_->scope()) || scope) << "Either scope should be set";
  lite::Scope *the_scope = scope ? scope : op_->scope();
  op_->Attach(op_desc, the_scope);
  // Recreate the kernels with the latest OpInfo.
  valid_kernels_.clear();
// Error message: if current kernel is not supported, WITH_EXTRA lib is
// suggested.
#ifndef LITE_BUILD_EXTRA
  std::string error_message =
      "Error: Please use Paddle-Lite lib with all ops, which is marked with "
      "`with_extra`. Current lib is of tiny_publish, in which only basic ops "
      "are included and we can not create operator '"
      << op_desc.Type()
      << "'.\n Two ways are suggested to get Paddle-Lite lib with all ops:\n   "
         " 1. Download pre-commit lib which is marked with `with_extra`.\n    "
         "2. Compile Paddle-Lite with command `--with_extra=ON`.";
#else
  std::string error_message =
      "Error: This model is not supported, because operator '" +
      op_desc.Type() + "' is not supported by Paddle-Lite.";
#endif
  if (!op_ || op_->op_info()->Type() != op_desc.Type()) {
    op_ = LiteOpRegistry::Global().Create(op_desc.Type());
    CHECK(op_) << error_message;
  }
  valid_kernels_ = op_->CreateKernels(valid_places);
}

void mir::Node::Stmt::ResetKernels(const std::vector<Place> &valid_places) {
  CHECK(op_) << "change valid place failed, not created op";
  valid_kernels_.clear();
  valid_kernels_ = op_->CreateKernels(valid_places);
}

mir::Node::Arg &mir::Node::AsArg(const std::string &name, int id) {
  auto &x = AsArg();
  x.name = name;
  x.id = id;
  return x;
}
mir::Node::Arg &mir::Node::AsArg(const std::string &name) {
  auto &x = AsArg();
  x.name = name;
  return x;
}
}  // namespace lite
}  // namespace paddle
