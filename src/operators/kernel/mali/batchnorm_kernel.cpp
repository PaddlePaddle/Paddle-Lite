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

#ifdef BATCHNORM_OP

#include "operators/kernel/batchnorm_kernel.h"
#ifdef PADDLE_MOBILE_MALI_GPU
#include "acl_operator.h"
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class AclBatchNormOp : public acl::ACLOperator {
 public:
  AclBatchNormOp() {
    this->force_bypass_acl_path_ = bypass_acl_class_layer & FLAGS_ENABLE_ACL_BN;
  }
  ~AclBatchNormOp() = default;
  AclBatchNormOp(const AclBatchNormOp&) = delete;
  AclBatchNormOp& operator=(const AclBatchNormOp&) = delete;
  AclBatchNormOp(AclBatchNormOp&&) = delete;
  AclBatchNormOp& operator=(AclBatchNormOp&&) = delete;

  acl::AclParameters& getargs() { return args; }
  void InitAclLayer(const BatchNormParam<DeviceType>& param) {
    setTargetHint(acl::TargetHint::OPENCL);
    arm_compute::TensorShape input_shape(args.in_cols, args.in_rows,
                                         args.in_depth, args.batch);
    arm_compute::TensorShape output_shape(args.out_cols, args.out_rows,
                                          args.out_depth, args.out_num);

    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_ = false;

    arm_compute::TensorShape mean_shape(args.in_depth);
    arm_compute::TensorShape var_shape = mean_shape;
    arm_compute::TensorShape beta_shape = mean_shape;
    arm_compute::TensorShape gamma_shape = mean_shape;

    //[width, height, IFM]
    new_tensor(input(), input_shape, args.input_data);
    //[width, height, OFM]
    new_tensor(output(), output_shape, args.output_data);

    new_tensor(mean(), mean_shape, args.mean_data);
    new_tensor(var(), var_shape, args.var_data);
    new_tensor(beta(), beta_shape, args.biases_data);
    new_tensor(gamma(), gamma_shape, args.weight_data);

    acl_configure(bn, this, args.epsilon);
  }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const BatchNormParam<DeviceType>& param) {
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
  void AclParametersByContext(const BatchNormParam<DeviceType>& param) {
    const Tensor* in_x = param.InputX();
    Tensor* out = param.OutputY();
    const Tensor* scale = param.InputScale();
    const Tensor* bias = param.InputBias();
    const Tensor* saved_mean = param.InputMean();
    const Tensor* saved_variance = param.InputVariance();

    const T* input_data = in_x->data<T>();
    T* output_data = out->mutable_data<T>();
    const T* weight_data = scale->data<T>();
    const T* bias_data = bias->data<T>();
    const T* mean_data = saved_mean->data<T>();
    const T* var_data = saved_variance->data<T>();

    float epsilon = param.Epsilon();

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;
    // args.weight_data = (void*)weight_data;
    // args.biases_data = (void*)bias_data;
    args.mean_data = (void*)mean_data;
    args.var_data = (void*)var_data;
    args.epsilon = epsilon;

    args.dim = in_x->dims().size();

    args.batch = in_x->dims()[0];
    args.in_depth = in_x->dims()[1];
    args.in_rows = in_x->dims()[2];
    args.in_cols = in_x->dims()[3];

    args.out_num = out->dims()[0];
    args.out_depth = out->dims()[1];
    args.out_rows = out->dims()[2];
    args.out_cols = out->dims()[3];

    args.weight_data = (void*)weight_data;
    args.biases_data = (void*)bias_data;

    // std::cout
    //  << "Out C: " <<  args.out_depth
    //  << " H: " << args.out_rows << " W: " << args.out_cols << "\n";
  }
  acl::AclParameters args;
};

template <>
bool BatchNormKernel<GPU_MALI, float>::Init(BatchNormParam<GPU_MALI>* param) {
  AclBatchNormOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclBatchNormOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    acl_op = new AclBatchNormOp<GPU_MALI, float>();
    this->SetAclOp((void*)acl_op, (void*)this);
  }
  if (acl_op->Bypass_acl(*param)) {
    std::cout << "init acl failed" << std::endl;
    return false;
  }
  return true;
}

template <>
void BatchNormKernel<GPU_MALI, float>::Compute(
    const BatchNormParam<GPU_MALI>& param) {
  std::cout << "init acl" << std::endl;
  AclBatchNormOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclBatchNormOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    return;
  }
  acl::AclParameters& args = acl_op->getargs();
  acl_op->RunAcl(args.input_data, args.output_data);
}

template class BatchNormKernel<GPU_MALI, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
#endif
