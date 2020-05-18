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

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/scope.h"
#include "lite/model_parser/cpp/op_desc.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {

// For registry factory.
struct Registry {
  void Touch() {}
};

namespace mir {
class Node;
class SSAGraph;
}

class OpInfo;

/**
 * The base class of an light-weight operators, currently just used in inference
 * to eliminate overhead of some operations in current framework.
 *
 * The Operator are designed as follows:
 * - it can has some members to hold the argument and some other computation
 * resources,
 * - it should act like a function call, no more logic included.
 */
class OpLite : public Registry {
 public:
  OpLite() = default;
  explicit OpLite(const std::string &type) : op_type_(type) {}
  explicit OpLite(const std::vector<Place> &valid_places)
      : valid_places_(valid_places) {}

  void SetValidPlaces(const std::vector<Place> &places) {
    VLOG(5) << "valid places " << valid_places_.size();
    valid_places_ = places;
  }
  const std::vector<Place> &valid_places() const { return valid_places_; }
  // Check the shape.
  virtual bool CheckShape() const { return true; }
  // Inference the outputs' shape.
  virtual bool InferShapeImpl() const { return true; }
  virtual bool InferShape();
  // Run this operator.
  virtual bool Run();
  // Indicate whether the Op runs only once or not
  virtual bool run_once() const { return false; }
  std::string Type() { return op_type_; }

  // Link the external execution environ to internal context.
  bool Attach(const cpp::OpDesc &opdesc, lite::Scope *scope);

  template <typename T>
  inline void AttachParam(T *param) {
    op_param_ = static_cast<T *>(param);
  }

  const OpInfo *op_info() const { return op_info_.get(); }
  OpInfo *mutable_op_info() { return op_info_.get(); }

  // Human-readable information.
  virtual std::string DebugString() const = 0;

  virtual std::string SerializedOpInfo() const { return "N/A"; }

  const Place &kernel_place() const { return kernel_place_; }

  // Create all the kernels for the valid targets.
  std::vector<std::unique_ptr<KernelBase>> CreateKernels(
      const std::vector<Place> &places, const std::string &kernel_type = "");

  lite::Scope *scope() { return scope_; }

  // Assign op param to kernel.
  virtual void AttachKernel(KernelBase *kernel) = 0;
  void SetKernel(std::vector<std::unique_ptr<KernelBase>> &kernels) {  // NOLINT
    kernel_ = std::move(kernels.front());
    kernel_->SetContext(
        ContextScheduler::Global().NewContext(kernel_->target()));
  }

  KernelBase *GetKernel() {  // NOLINT
    return kernel_.get();
  }

  // Attach input variable from scope by op_desc and input name
  void AttachInput(const cpp::OpDesc &op_desc,
                   lite::Scope *scope,
                   const std::string &input_name,
                   bool is_dispensable,
                   lite::Tensor **input_var);

  // Attach output variable from scope by op_desc and output name
  void AttachOutput(const cpp::OpDesc &op_desc,
                    lite::Scope *scope,
                    const std::string &output_name,
                    bool is_dispensable,
                    lite::Tensor **output_var);

  virtual ~OpLite() = default;

 protected:
  // Attach it with the runtime environment.
  virtual bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) = 0;

  // Specify the kernel to run by default. This will specify the value of
  // `kernel_place_`.
  virtual void StaticPickKernel(const std::vector<Place> &valid_targets) {
    auto kernels = CreateKernels(valid_targets);
    kernel_ = std::move(kernels.front());
  }

  // Wait until all the inputs' events are ready.
  void SyncInputEvents() {}

  // Record the output events, and that will tell all the dependent operators
  // some inputs are ready.
  void RecordOutputEvents() {}

  const Tensor *GetTensor(lite::Scope *scope, const std::string &name) const;
  Tensor *GetMutableTensor(lite::Scope *scope, const std::string &name) const;

  friend class mir::Node;
  friend class mir::SSAGraph;

 protected:
  // some helper functions.
  template <typename T>
  const T *GetVar(Scope *scope, const std::string &name) {
    auto *var = scope->FindVar(name);
    CHECK(var) << "No var found for " << name;
    return &var->Get<T>();
  }
  template <typename T>
  T *GetMutableVar(Scope *scope, const std::string &name) {
    auto *var = scope->FindVar(name);
    CHECK(var) << "No var found for " << name;
    return var->GetMutable<T>();
  }

 protected:
  lite::Scope *scope_{nullptr};
  std::unique_ptr<KernelBase> kernel_;
  std::string op_type_;
  std::vector<Place> valid_places_;
  Place kernel_place_{TARGET(kHost), PRECISION(kFloat)};
  std::unique_ptr<OpInfo> op_info_;
  // todo: it's prefered to combine last_input_shapes and
  // last_input_lods into a single hash value to decrease
  // memory usage.
  std::vector<DDimLite> last_input_shapes{};
  std::vector<std::vector<std::vector<uint64_t>>> last_input_lods{};
  std::vector<DDimLite> last_output_shapes{};
  std::vector<std::vector<std::vector<uint64_t>>> last_output_lods{};
  mutable operators::ParamBase *op_param_{nullptr};

 private:
  // Infer Shape according to memory, if current input shapes are consistent
  // with that of previous inputs, output shapes of last time will be reused.
  bool InferShapeWithCache();
};

/*
 * Operator Information, such as some description. It will be shared by all the
 * kernels of the same operator.
 */
class OpInfo : public cpp::OpDesc {
 public:
  OpInfo(const OpInfo &) = default;
  explicit OpInfo(const cpp::OpDesc &other) : cpp::OpDesc(other) {}

  // Collect all the input variable's name.
  std::vector<std::string> input_names() const {
    std::vector<std::string> res;
    for (auto &param : InputArgumentNames()) {
      for (auto &x : Input(param)) {
        res.push_back(x);
      }
    }
    return res;
  }

  // Collect all the output variable's name.
  std::vector<std::string> output_names() const {
    std::vector<std::string> res;
    for (auto &param : OutputArgumentNames()) {
      for (auto &x : Output(param)) {
        res.push_back(x);
      }
    }
    return res;
  }

  std::vector<std::string> input_argnames() const {
    return InputArgumentNames();
  }

  std::vector<std::string> output_argnames() const {
    return OutputArgumentNames();
  }

  bool GetInputArgname(const std::string &value_name, std::string *out) const {
    for (auto &item : inputs_) {
      auto it = std::find(item.second.begin(), item.second.end(), value_name);
      if (it != item.second.end()) {
        *out = item.first;
        return true;
      }
    }
    return false;
  }
  bool GetOutputArgname(const std::string &value_name, std::string *out) const {
    for (auto &item : outputs_) {
      auto it = std::find(item.second.begin(), item.second.end(), value_name);
      if (it != item.second.end()) {
        *out = item.first;
        return true;
      }
    }
    return false;
  }

  // For the input variable name, find the index of the corresponding
  // input argname
  bool GetInputIndex(const std::string &value_name, int *out) const {
    for (auto &item : inputs_) {
      auto it = std::find(item.second.begin(), item.second.end(), value_name);
      if (it != item.second.end()) {
        *out = it - item.second.begin();
        return true;
      }
    }
    return false;
  }

  // For the output variable name, find the index of the corresponding
  // output argname
  bool GetOutputIndex(const std::string &value_name, int *out) const {
    for (auto &item : outputs_) {
      auto it = std::find(item.second.begin(), item.second.end(), value_name);
      if (it != item.second.end()) {
        *out = it - item.second.begin();
        return true;
      }
    }
    return false;
  }

  void UpdateAllInputs(const std::string &from, const std::string &to) {
    for (auto &item : inputs_) {
      for (auto &var : item.second) {
        if (var == from) var = to;
      }
    }
  }

  void UpdateAllOutputs(const std::string &from, const std::string &to) {
    for (auto &item : outputs_) {
      for (auto &var : item.second) {
        if (var == from) var = to;
      }
    }
  }
};

}  // namespace lite
}  // namespace paddle
