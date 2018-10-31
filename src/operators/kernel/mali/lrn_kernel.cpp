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

#ifdef LRN_OP

#pragma once

#include "operators/kernel/lrn_kernel.h"
#ifdef PADDLE_MOBILE_MALI_GPU
#include "acl_operator.h"
#include "framework/operator.h"
#include "operators/kernel/central-arm-func/lrn_arm_func.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class AclLrnOp : public acl::ACLOperator {
 public:
  AclLrnOp() {
    this->force_bypass_acl_path_ =
        bypass_acl_class_layer & FLAGS_ENABLE_ACL_LRN;
  }
  ~AclLrnOp() = default;
  AclLrnOp(const AclLrnOp&) = delete;
  AclLrnOp& operator=(const AclLrnOp&) = delete;
  AclLrnOp(AclLrnOp&&) = delete;
  AclLrnOp& operator=(AclLrnOp&&) = delete;

  acl::AclParameters& getargs() { return args; }
  void InitAclLayer(const LrnParam<DeviceType>& param) {
    setTargetHint(acl::TargetHint::OPENCL);
    arm_compute::TensorShape shape(args.in_cols, args.in_rows, args.in_depth);

    if (is_operator_init_done(shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_ = false;

    arm_compute::NormalizationLayerInfo norm_info(
        arm_compute::NormType::CROSS_MAP, args.nsize, args.alpha, args.beta,
        args.knorm);

    //[width, height, IFM]
    new_tensor(input(), shape, args.input_data);
    //[width, height, OFM]
    new_tensor(output(), shape, args.output_data);

    acl_configure(lrn, this, norm_info);
  }

  void Set_bypass(bool bypass) { args.is_bypass = bypass; }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const LrnParam<DeviceType>& param) {
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
  void AclParametersByContext(const LrnParam<DeviceType>& param) {
    const Tensor* in_x = param.InputX();
    Tensor* out = param.Out();

    int n = param.N();
    T alpha = param.Alpha();
    T beta = param.Beta();
    T k = param.K();

    const T* input_data = in_x->data<T>();
    T* output_data = out->mutable_data<T>();

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;

    args.nsize = n;
    args.alpha = alpha;
    args.beta = beta;
    args.knorm = k;

    // NCHW
    args.batch = in_x->dims()[0];
    args.in_depth = in_x->dims()[1];
    args.in_rows = in_x->dims()[2];
    args.in_cols = in_x->dims()[3];
    // std::cout
    //  << "Out C: " <<  args.out_depth
    //  << " H: " << args.out_rows << " W: " << args.out_cols << "\n";
  }
  acl::AclParameters args;
};

template <>
bool LrnKernel<GPU_MALI, float>::Init(LrnParam<GPU_MALI>* param) {
  AclLrnOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclLrnOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    acl_op = new AclLrnOp<GPU_MALI, float>();
    this->SetAclOp((void*)acl_op, (void*)this);
  }
  if (acl_op->Bypass_acl(*param)) {
    acl_op->Set_bypass(true);
    std::cout << "init acl failed" << std::endl;
    return true;
  }
  return true;
}

template <>
void LrnKernel<GPU_MALI, float>::Compute(const LrnParam<GPU_MALI>& param) {
  std::cout << "init acl" << std::endl;
  AclLrnOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclLrnOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    return;
  }
  acl::AclParameters& args = acl_op->getargs();
  if (args.is_bypass) {
    std::cout << "bypass op" << std::endl;
    LrnCompute<float>(param);
    return;
  }
  const float* input_data = (const float*)args.input_data;
  const float* output_data = (const float*)args.output_data;
  for (int n = 0; n < args.batch; ++n) {
    acl_op->RunAcl((void*)input_data, (void*)output_data);
    input_data += args.in_depth * args.in_cols * args.in_rows;
    output_data += args.in_depth * args.in_cols * args.in_rows;
  }
}

template class LrnKernel<GPU_MALI, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
#endif
