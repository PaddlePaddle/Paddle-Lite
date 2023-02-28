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
#include "lite/model_parser/cpp_desc.h"
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
  // Infer the outputs's data type during opt period
  virtual bool InferType() {
    LOG(FATAL) << "Error! " << op_type_
               << "::InferType() function must be registered for op "
               << op_type_;
    return false;
  }
  // Run this operator.
  virtual bool Run();
  // Indicate whether the Op runs only once or not
  virtual bool run_once() const { return false; }
  std::string Type() const { return op_type_; }
#ifdef LITE_WITH_PROFILE
  virtual void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {}
#endif

  // Link the external execution environ to internal context.
  bool Attach(const cpp::OpDesc &opdesc, lite::Scope *scope);

  virtual bool AttachInput(const cpp::OpDescWrite &opdesc, lite::Scope *scope) {
    return false;
  }
#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
  bool Attach(const cpp::OpDescWrite &opdesc, lite::Scope *scope);
#endif
  const OpInfo *op_info() const { return op_info_.get(); }
  OpInfo *mutable_op_info() { return op_info_.get(); }

  // Human-readable information.
  virtual std::string DebugString() const = 0;

  virtual std::string SerializedOpInfo() const { return "N/A"; }

  const Place &kernel_place() const { return kernel_place_; }

  // Create all the kernels for the valid targets.
  std::vector<std::unique_ptr<KernelBase>> CreateKernels(
      const std::vector<Place> &places, const std::string &kernel_type = "");

  Scope *scope() { return scope_; }

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
#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
  virtual bool AttachImpl(const cpp::OpDescWrite &opdesc, lite::Scope *scope) {
    return false;
  }
#endif
  virtual bool InferShapeWithCache() const { return false; }
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
  Scope *scope_{nullptr};
  std::unique_ptr<KernelBase> kernel_;
  std::string op_type_;
  std::vector<Place> valid_places_;
  Place kernel_place_{TARGET(kHost), PRECISION(kFloat)};
  std::unique_ptr<OpInfo> op_info_;
  // Infer Shape according to memory, if current input shapes are consistent
  // with that of previous inputs, output shapes of last time will be reused.
  std::vector<const Tensor *> input_tensor_ptrs_cache_{};
  std::vector<Tensor *> output_tensor_ptrs_cache_{};

 private:
  // todo: it's prefered to combine last_input_shapes and
  // last_input_lods into a single hash value to decrease
  // memory usage.
  std::vector<DDimLite> last_input_shapes_{};
  std::vector<LoD> last_input_lods_{};
  std::vector<DDimLite> last_output_shapes_{};
  std::vector<LoD> last_output_lods_{};
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

  void UpdateAllInputs(const std::string &from, const std::string &to) {
    for (auto &item : *mutable_inputs()) {
      for (auto &var : item.second) {
        if (var == from) var = to;
      }
    }
  }

  void UpdateAllOutputs(const std::string &from, const std::string &to) {
    for (auto &item : *mutable_outputs()) {
      for (auto &var : item.second) {
        if (var == from) var = to;
      }
    }
  }

  bool GetInputArgname(const std::string &value_name, std::string *out) const;
  bool GetOutputArgname(const std::string &value_name, std::string *out) const;

  bool GetInputIndex(const std::string &input_name, int *out) const;
  bool GetOutputIndex(const std::string &output_name, int *out) const;

  // If a quantized op has two input argname (X, Y) and one output
  // argname (Out). The scales of input argname X are saved in op desc as
  // (X0_scale, scale_value_0), (X1_scale, scale_value_1)...
  // The following APIs get or set the quantized scale in op_desc.
  // If use the input or output name, the is_scale_name should be false.
  // If use the scale_name such as (X0_scale, scale_value_0),
  // the is_scale_name should be true.
  bool HasInputScale(const std::string &name, bool is_scale_name = false) const;
  bool HasOutputScale(const std::string &name,
                      bool is_scale_name = false) const;

  void SetInputScale(const std::string &input_name,
                     const std::vector<float> &scale_value,
                     bool is_scale_name = false);
  void SetOutputScale(const std::string &output_name,
                      const std::vector<float> &scale_value,
                      bool is_scale_name = false);

  // For conv2d, depthwise_conv2d and mul, the scale of weight are a vector.
  // Otherwise, all input and output scales are scalar, but we save these
  // as vecotr.
  std::vector<float> GetInputScale(const std::string &name,
                                   bool is_scale_name = false) const;
  std::vector<float> GetOutputScale(const std::string &name,
                                    bool is_scale_name = false) const;
};

}  // namespace lite
}  // namespace paddle
