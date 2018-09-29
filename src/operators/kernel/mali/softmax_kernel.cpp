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

#ifdef SOFTMAX_OP

#pragma once

#include "operators/kernel/softmax_kernel.h"
#ifdef PADDLE_MOBILE_MALI_GPU
#include "acl_operator.h"
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class AclSoftmaxOp : public acl::ACLOperator {
 public:
  AclSoftmaxOp() {
    this->force_bypass_acl_path_ =
        bypass_acl_class_layer & FLAGS_ENABLE_ACL_SOFTMAX;
  }
  ~AclSoftmaxOp() = default;
  AclSoftmaxOp(const AclSoftmaxOp&) = delete;
  AclSoftmaxOp& operator=(const AclSoftmaxOp&) = delete;
  AclSoftmaxOp(AclSoftmaxOp&&) = delete;
  AclSoftmaxOp& operator=(AclSoftmaxOp&&) = delete;

  acl::AclParameters& getargs() { return args; }
  void InitAclLayer(const SoftmaxParam<DeviceType>& param) {
    setTargetHint(acl::TargetHint::OPENCL);
    arm_compute::TensorShape shape(args.in_depth, args.batch);

    if (is_operator_init_done(shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_ = false;

    //[width, height, IFM]
    new_tensor(input(), shape, args.input_data);
    //[width, height, OFM]
    new_tensor(output(), shape, args.output_data);

    acl_configure(softmax, this, NULL);
  }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const SoftmaxParam<DeviceType>& param) {
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
  void AclParametersByContext(const SoftmaxParam<DeviceType>& param) {
    const framework::Tensor* in_x = param.InputX();
    framework::Tensor* out = param.Out();
    auto x_dims = in_x->dims();
    out->Resize(x_dims);

    const T* input_data = in_x->data<T>();
    T* output_data = out->data<T>();

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;

    // NCHW
    args.batch = in_x->dims()[0];
    args.in_depth = in_x->dims()[1];

    args.out_num = out->dims()[0];

    // std::cout
    //  << "Out C: " <<  args.out_depth
    //  << " H: " << args.out_rows << " W: " << args.out_cols << "\n";
  }
  acl::AclParameters args;
};

template <>
bool SoftmaxKernel<GPU_MALI, float>::Init(SoftmaxParam<GPU_MALI>* param) {
  AclSoftmaxOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclSoftmaxOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    acl_op = new AclSoftmaxOp<GPU_MALI, float>();
    this->SetAclOp((void*)acl_op, (void*)this);
  }
  if (acl_op->Bypass_acl(*param)) {
    std::cout << "init acl failed" << std::endl;
    return false;
  }
  return true;
}

template <>
void SoftmaxKernel<GPU_MALI, float>::Compute(
    const SoftmaxParam<GPU_MALI>& param) {
  std::cout << "init acl" << std::endl;
  AclSoftmaxOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclSoftmaxOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    return;
  }
  acl::AclParameters& args = acl_op->getargs();
  const float* input_data = (const float*)args.input_data;
  const float* output_data = (const float*)args.output_data;

  for (int n = 0; n < args.out_num; ++n) {
    acl_op->RunAcl((void*)input_data, (void*)output_data);
    input_data += args.in_depth;
    output_data += args.in_depth;
  }
}

template class SoftmaxKernel<GPU_MALI, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
#endif
