/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef RELU_OP

#pragma once

#include "operators/kernel/relu_kernel.h"
#ifdef PADDLE_MOBILE_MALI_GPU
#include "acl_operator.h"
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class AclReluOp : public acl::ACLOperator {
 public:
  AclReluOp() {
    this->force_bypass_acl_path_ =
        bypass_acl_class_layer & FLAGS_ENABLE_ACL_RELU;
  }
  ~AclReluOp() = default;
  AclReluOp(const AclReluOp&) = delete;
  AclReluOp& operator=(const AclReluOp&) = delete;
  AclReluOp(AclReluOp&&) = delete;
  AclReluOp& operator=(AclReluOp&&) = delete;

  acl::AclParameters& getargs() { return args; }
  void InitAclLayer(const ReluParam<DeviceType>& param) {
    setTargetHint(acl::TargetHint::OPENCL);
    arm_compute::TensorShape input_shape(args.in_cols, args.in_rows,
                                         args.in_depth, args.batch);
    arm_compute::TensorShape output_shape(args.in_cols, args.in_rows,
                                          args.in_depth, args.out_num);
    // arm_compute::TensorShape weights_shape(
    // args.filter_cols, args.filter_rows, args.in_depth, args.out_depth);
    // arm_compute::TensorShape biases_shape(args.out_depth);
    arm_compute::ActivationLayerInfo::ActivationFunction type;
    type = arm_compute::ActivationLayerInfo::ActivationFunction::RELU;

    arm_compute::ActivationLayerInfo act_info(type);

    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_ = false;

    //[width, height, IFM]
    new_tensor(input(), input_shape, args.input_data);
    //[width, height, OFM]
    new_tensor(output(), output_shape, args.output_data);

    acl_configure(activation, this, act_info);
  }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const ReluParam<DeviceType>& param) {
    bool bypass_acl = false;
    AclParametersByContext(param);
    InitAclLayer(param);
    // for performance, more groups impact GPU performance
    if (this->force_bypass_acl_path_) {
      bypass_acl = true;
    }
    return bypass_acl;
  }

 private:
  void AclParametersByContext(const ReluParam<DeviceType>& param) {
    const auto* input_x = param.InputX();
    auto* out = param.Out();

    const T* input_data = input_x->data<T>();
    T* output_data = out->mutable_data<T>();

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;

    args.batch = input_x->dims()[0];
    args.in_depth = input_x->dims()[1];
    args.in_rows = input_x->dims()[2];
    args.in_cols = input_x->dims()[3];
    args.out_num = out->dims()[0];
  }
  acl::AclParameters args;
};

template <>
bool ReluKernel<GPU_MALI, float>::Init(ReluParam<GPU_MALI>* param) {
  AclReluOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclReluOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    acl_op = new AclReluOp<GPU_MALI, float>();
    this->SetAclOp((void*)acl_op, (void*)this);
  }
  if (acl_op->Bypass_acl(*param)) {
    std::cout << "init acl failed" << std::endl;
    return false;
  }
  return true;
}

template <>
void ReluKernel<GPU_MALI, float>::Compute(const ReluParam<GPU_MALI>& param) {
  std::cout << "init acl" << std::endl;
  AclReluOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclReluOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    return;
  }
  acl::AclParameters& args = acl_op->getargs();
  acl_op->RunAcl(args.input_data, args.output_data);
}

template class ReluKernel<GPU_MALI, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
#endif
